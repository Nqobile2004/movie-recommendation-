import streamlit as st
import pandas as pd
from collaborative import SVDRecommender
from content_based import ContentRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")
st.write("SVD Collaborative + Content-Based Filtering")

# Load sample data
@st.cache_data
def load_data():
    ratings = pd.read_csv("sample_data/ratings.csv")
    movies = pd.read_csv("sample_data/movies.csv")
    return ratings, movies

try:
    ratings_df, movies_df = load_data()
    st.success(f"Loaded {len(movies_df)} movies")

    user_id = st.number_input("Enter User ID", min_value=1, value=1)
    
    if st.button("Recommend"):
        model = SVDRecommender(n_factors=20)
        model.fit(ratings_df)
        recs = model.recommend(user_id, n=10)
        st.write("Recommended for you:")
        st.dataframe(recs)

except Exception as e:
    st.error(f"Need sample_data: {e}")
