import streamlit as st
import pandas as pd
from collaborative import SVDRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")
st.write("SVD Collaborative Filtering - Live Demo")

@st.cache_data
def load_data():
    try:
        ratings = pd.read_csv("sample_data/ratings.csv")
        movies = pd.read_csv("sample_data/movies.csv")
        return ratings, movies
    except:
        return None, None

ratings_df, movies_df = load_data()

if ratings_df is None:
    st.warning("Add sample_data folder with ratings.csv and movies.csv, or I can create dummy data for demo")
    # create dummy data for demo to make it work
    import numpy as np
    ratings_df = pd.DataFrame({
        'userId': np.random.randint(1,100, 500),
        'movieId': np.random.randint(1,50, 500),
        'rating': np.random.randint(1,6, 500)
    })
    movies_df = pd.DataFrame({
        'movieId': range(1,51),
        'title': [f"Movie {i}" for i in range(1,51)]
    })

st.success(f"Loaded {len(movies_df)} movies")

user_id = st.number_input("Enter User ID", min_value=1, max_value=99, value=1)

if st.button("Get Recommendations"):
    model = SVDRecommender(n_factors=20)
    model.fit(ratings_df)
    recs = model.recommend(user_id, n=10)
    st.write("Recommended movie IDs for you:")
    st.dataframe(recs)
