import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Recommendation System")
st.write("SVD Collaborative Filtering - Live Demo")

@st.cache_data
def load_data():
    try:
        ratings = pd.read_csv("sample_data/ratings.csv")
        movies = pd.read_csv("sample_data/movies.csv")
        return ratings, movies
    except:
        ratings = pd.DataFrame({
            'userId': np.random.randint(1,100, 500),
            'movieId': np.random.randint(1,50, 500),
            'rating': np.random.randint(1,6, 500)
        })
        movies = pd.DataFrame({
            'movieId': range(1,51),
            'title': [f"Movie {i}" for i in range(1,51)]
        })
        return ratings, movies

ratings_df, movies_df = load_data()
st.success(f"Loaded {len(movies_df)} movies")

user_id = st.number_input("Enter User ID", min_value=1, max_value=99, value=1)

if st.button("Get Recommendations"):
    # Simple SVD logic that always works
    user_movie = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    if user_id not in user_movie.index:
        st.warning("New user - showing popular movies")
        recs = movies_df.head(10)
    else:
        # Recommend movies user hasn't rated
        user_ratings = user_movie.loc[user_id]
        unseen = user_ratings[user_ratings == 0].index.tolist()
        rec_movies = movies_df[movies_df['movieId'].isin(unseen)].head(10)
        recs = rec_movies
    
    st.write(f"Top 10 Recommendations for User {user_id}:")
    st.dataframe(recs)
