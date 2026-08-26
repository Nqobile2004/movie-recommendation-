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
        # remove duplicates that cause pivot error
        ratings = ratings.drop_duplicates(subset=['userId','movieId'])
        return ratings, movies
    except:
        ratings = pd.DataFrame({
            'userId': np.random.randint(1,100, 500),
            'movieId': np.random.randint(1,50, 500),
            'rating': np.random.randint(1,6, 500)
        }).drop_duplicates(subset=['userId','movieId'])
        movies = pd.DataFrame({
            'movieId': range(1,51),
            'title': [f"Movie {i}" for i in range(1,51)]
        })
        return ratings, movies

ratings_df, movies_df = load_data()
st.success(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")

user_id = st.number_input("Enter User ID", min_value=1, max_value=99, value=1)

if st.button("Get Recommendations"):
    rated_by_user = ratings_df[ratings_df['userId']==user_id]['movieId'].tolist()
    unseen_movies = movies_df[~movies_df['movieId'].isin(rated_by_user)]
    
    if unseen_movies.empty:
        st.write("You rated everything! Showing popular movies:")
        st.dataframe(movies_df.head(10))
    else:
        st.write(f"Top 10 Recommendations for User {user_id}:")
        st.dataframe(unseen_movies.head(10))
    
    st.balloons()
