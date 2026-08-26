"""
Content-based filtering: recommends movies similar to ones a user rated highly,
based on genre overlap (TF-IDF + cosine similarity).

Doesn't need other users' data at all -- this is what makes it useful for the
"cold start" problem (brand new movies with no ratings yet).
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, movies_df):
        self.movies = movies_df.reset_index(drop=True)
        # Treat genre string "Action|Comedy" as a mini "document" of words
        self.movies["genres_text"] = self.movies["genres"].str.replace("|", " ", regex=False)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.movies["genres_text"])
        self.similarity_matrix = cosine_similarity(tfidf_matrix)

        self.movie_id_to_idx = {
            mid: idx for idx, mid in enumerate(self.movies["movieId"])
        }

    def similar_movies(self, movie_id, n=10):
        """Given one movie, find the n most similar movies by genre."""
        if movie_id not in self.movie_id_to_idx:
            return pd.DataFrame()
        idx = self.movie_id_to_idx[movie_id]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:n]

        result_idx = [s[0] for s in scores]
        result = self.movies.iloc[result_idx][["movieId", "title", "genres"]].copy()
        result["similarity"] = [s[1] for s in scores]
        return result

    def recommend_for_user(self, user_ratings, n=10):
        """
        user_ratings: DataFrame with columns [movieId, rating] for one user.
        Builds a weighted profile of the user's taste from their highly-rated
        movies, then finds unrated movies closest to that profile.
        """
        liked = user_ratings[user_ratings["rating"] >= 4.0]
        if liked.empty:
            liked = user_ratings.nlargest(5, "rating")

        rated_movie_ids = set(user_ratings["movieId"])
        candidate_scores = {}

        for _, row in liked.iterrows():
            mid = row["movieId"]
            if mid not in self.movie_id_to_idx:
                continue
            idx = self.movie_id_to_idx[mid]
            for other_idx, sim in enumerate(self.similarity_matrix[idx]):
                other_mid = self.movies.iloc[other_idx]["movieId"]
                if other_mid in rated_movie_ids:
                    continue
                # weight similarity by how much the user liked the seed movie
                candidate_scores[other_mid] = candidate_scores.get(other_mid, 0) + sim * row["rating"]

        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        rec_ids = [mid for mid, _ in ranked]
        result = self.movies[self.movies["movieId"].isin(rec_ids)][["movieId", "title", "genres"]].copy()
        result["score"] = result["movieId"].map(dict(ranked))
        return result.sort_values("score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    cb = ContentBasedRecommender(movies)

    print("Movies similar to Movie 1:")
    print(cb.similar_movies(movie_id=1, n=5), "\n")

    user1_ratings = ratings[ratings["userId"] == 1][["movieId", "rating"]]
    print(f"User 1 has rated {len(user1_ratings)} movies")
    print("Content-based recommendations for User 1:")
    print(cb.recommend_for_user(user1_ratings, n=5))
