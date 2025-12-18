import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="SKU Substitution System", layout="wide")
st.title("Substitute Products for Out-of-Stock SKUs")

st.write(
    "Search for an **unavailable product** and view "
    "**similar, available substitute products**."
)

# --------------------------------------------------
# Load & prepare data
# --------------------------------------------------
def load_data():
    products_df = pd.read_csv(
        "products.csv",
        usecols=["asin", "title", "breadcrumbs", "availability", "about_item", "price_value"]
    )

    # Out-of-stock flag
    products_df["out_of_stock"] = products_df["availability"].isin([
        "Currently unavailable.",
        "Temporarily out of stock."
    ])

    # ----------------------------
    # Main category (Men / Baby / Women)
    # ----------------------------
    def get_main_category(breadcrumbs):
        parts = breadcrumbs.split("›")
        return parts[1].strip() if len(parts) > 1 else "Unknown"

    products_df["main_category"] = products_df["breadcrumbs"].apply(get_main_category)

    # ----------------------------
    # Product type (Jeans, Pants, Shirts, etc.)
    # ----------------------------
    STYLE_KEYWORDS = {
        "casual", "formal", "party", "athletic", "sport", "sportswear",
        "slim", "regular", "relaxed", "classic", "fashion", "daily"
    }

    def get_product_type(breadcrumbs):
        parts = [p.strip().lower() for p in breadcrumbs.split("›")]
        if len(parts) < 2:
            return "unknown"
        if parts[-1] in STYLE_KEYWORDS:
            return parts[-2]          # Pants from "Pants › Casual"
        return parts[-1]              # Jeans from "… › Jeans"

    products_df["product_type"] = products_df["breadcrumbs"].apply(get_product_type)

    # ----------------------------
    # Ratings
    # ----------------------------
    reviews_df = pd.read_csv(
        "reviews.csv",
        usecols=["productASIN", "rating"]
    )

    ratings_agg = (
        reviews_df
        .groupby("productASIN", as_index=False)["rating"]
        .mean()
        .rename(columns={"productASIN": "asin", "rating": "avg_rating"})
    )

    products_df = products_df.merge(ratings_agg, on="asin", how="inner")

    # Fill missing text
    products_df["about_item"] = products_df["about_item"].fillna("")
    products_df["breadcrumbs"] = products_df["breadcrumbs"].fillna("")

    # Combined text for similarity
    products_df["combined_text"] = (
        products_df["title"] + " " +
        products_df["breadcrumbs"] + " " +
        products_df["about_item"]
    )

    return products_df


@st.cache_resource
def build_similarity(products_df):
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000
    )

    tfidf_matrix = tfidf.fit_transform(products_df["combined_text"])

    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(
        products_df[["price_value", "avg_rating"]]
    )

    final_vectors = hstack([tfidf_matrix, numeric_features])

    similarity_matrix = cosine_similarity(final_vectors)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=products_df["asin"],
        columns=products_df["asin"]
    )

    return similarity_df


# --------------------------------------------------
# Recommendation function
# --------------------------------------------------
def recommend_substitutes(products_df, similarity_df, retired_asin, top_n=3, price_tol=0.3):

    retired_row = products_df.loc[products_df["asin"] == retired_asin].iloc[0]
    retired_price = retired_row["price_value"]
    retired_cat = retired_row["main_category"]
    retired_type = retired_row["product_type"]

    # 🔒 STRONG CONSTRAINTS
    candidates = products_df[
        (~products_df["out_of_stock"]) &
        (products_df["main_category"] == retired_cat) &
        (products_df["product_type"] == retired_type) &   # 👈 FIX FOR JEANS vs TOPS
        (products_df["price_value"].between(
            retired_price * (1 - price_tol),
            retired_price * (1 + price_tol)
        ))
    ]

    if candidates.empty:
        return pd.DataFrame()

    scores = similarity_df.loc[retired_asin, candidates["asin"]]
    scores = scores.sort_values(ascending=False).head(top_n)

    recs = (
        scores
        .rename("similarity_score")
        .reset_index()
        .rename(columns={"index": "asin"})
    )

    recs = recs.merge(
        products_df[["asin", "title", "price_value", "avg_rating"]],
        on="asin",
        how="left"
    )

    recs["Price Diff (%)"] = (
        (recs["price_value"] - retired_price) / retired_price * 100
    ).round(1)

    return recs


# --------------------------------------------------
# Load data & model
# --------------------------------------------------
products_df = load_data()
similarity_df = build_similarity(products_df)

# --------------------------------------------------
# Sidebar: unavailable products only
# --------------------------------------------------
oos_df = products_df[products_df["out_of_stock"]].copy()

price_tol = st.sidebar.slider(
    "Price tolerance (%)",
    min_value=10,
    max_value=50,
    value=30
) / 100

selected_asin = st.sidebar.selectbox(
    "Search unavailable products",
    options=oos_df["asin"],
    format_func=lambda x: oos_df.loc[oos_df["asin"] == x, "title"].values[0]
)

# --------------------------------------------------
# Display selected product
# --------------------------------------------------
selected_row = oos_df[oos_df["asin"] == selected_asin].iloc[0]

st.subheader("Out-of-Stock Product")
st.write(f"**Title:** {selected_row['title']}")
st.write(f"**Price:** ${selected_row['price_value']:.2f}")
st.write(f"**Rating:** {selected_row['avg_rating']:.2f}")
st.write(f"**Main Category:** {selected_row['main_category']}")
st.write(f"**Product Type:** {selected_row['product_type']}")
st.write(f"**Availability:** {selected_row['availability']}")

# --------------------------------------------------
# Display recommendations
# --------------------------------------------------
st.subheader("✅ Recommended Substitute Products")

recs = recommend_substitutes(
    products_df,
    similarity_df,
    selected_asin,
    top_n=3,
    price_tol=price_tol
)

if recs.empty:
    st.warning("No suitable substitutes found within the selected constraints.")
else:
    recs_display = recs.rename(columns={
        "title": "Title",
        "price_value": "Price",
        "avg_rating": "Rating",
        "similarity_score": "Similarity Score"
    })

    recs_display["Price"] = recs_display["Price"].round(2)
    recs_display["Similarity Score"] = recs_display["Similarity Score"].round(2)

    st.dataframe(
        recs_display[["Title", "Price", "Rating", "Similarity Score", "Price Diff (%)"]],
        use_container_width=True
    )
