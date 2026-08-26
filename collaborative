"""
Collaborative filtering via matrix factorization (SVD).

Core idea: the user-movie rating matrix is huge and mostly empty (sparse).
SVD approximates it as the product of two much smaller matrices --
one describing each user as a vector of "latent taste factors", and one
describing each movie as a vector of "latent trait factors". Multiplying
a user's vector by a movie's vector predicts the rating.

We use sklearn's TruncatedSVD here (no extra dependencies needed). For
production work, the `scikit-surprise` or `implicit` libraries offer more
specialized solvers (e.g. proper SGD-based SVD, ALS) -- worth trying once
this baseline works.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


class SVDRecommender:
    def __init__(self, n_factors=20):
        self.n_factors = n_factors
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)

    def fit(self, ratings_df):
        self.ratings_df = ratings_df
        self.user_ids = sorted(ratings_df["userId"].unique())
        self.movie_ids = sorted(ratings_df["movieId"].unique())
        self.user_id_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_id_to_idx = {m: i for i, m in enumerate(self.movie_ids)}

        n_users = len(self.user_ids)
        n_movies = len(self.movie_ids)

        # Build the dense user x movie matrix. Missing ratings filled with
        # each user's mean rating -- a simple but effective way to handle
        # sparsity for SVD (which doesn't like raw zeros, since 0 looks
        # like "hated it" rather than "didn't rate it").
        matrix = np.zeros((n_users, n_movies))
        user_means = ratings_df.groupby("userId")["rating"].mean()
        global_mean = ratings_df["rating"].mean()

        for u in self.user_ids:
            matrix[self.user_id_to_idx[u], :] = user_means.get(u, global_mean)

        for _, row in ratings_df.iterrows():
            u_idx = self.user_id_to_idx[row["userId"]]
            m_idx = self.movie_id_to_idx[row["movieId"]]
            matrix[u_idx, m_idx] = row["rating"]

        self.global_mean = global_mean
        self.user_means = user_means

        # Mean-center each row before factorizing -- standard practice,
        # keeps the SVD focused on *deviations* from a user's average taste
        # rather than wasting factors just re-learning "user 5 rates high".
        self.row_means = matrix.mean(axis=1, keepdims=True)
        centered = matrix - self.row_means

        user_factors = self.svd.fit_transform(centered)
        movie_factors = self.svd.components_.T

        self.predicted_matrix = user_factors @ movie_factors.T + self.row_means
        self.predicted_matrix = np.clip(self.predicted_matrix, 0.5, 5.0)
        return self

    def predict(self, user_id, movie_id):
        if user_id not in self.user_id_to_idx or movie_id not in self.movie_id_to_idx:
            return self.global_mean
        u_idx = self.user_id_to_idx[user_id]
        m_idx = self.movie_id_to_idx[movie_id]
        return self.predicted_matrix[u_idx, m_idx]

    def recommend_for_user(self, user_id, movies_df, n=10):
        if user_id not in self.user_id_to_idx:
            return pd.DataFrame()
        u_idx = self.user_id_to_idx[user_id]
        already_rated = set(
            self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"]
        )

        predictions = []
        for m_idx, movie_id in enumerate(self.movie_ids):
            if movie_id in already_rated:
                continue
            predictions.append((movie_id, self.predicted_matrix[u_idx, m_idx]))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top = predictions[:n]

        rec_ids = [mid for mid, _ in top]
        result = movies_df[movies_df["movieId"].isin(rec_ids)][["movieId", "title", "genres"]].copy()
        result["predicted_rating"] = result["movieId"].map(dict(top))
        return result.sort_values("predicted_rating", ascending=False).reset_index(drop=True)


def popularity_baseline(ratings_df, movies_df, min_ratings=5, n=10):
    """Simple baseline: highest average rating among movies with enough votes."""
    stats = ratings_df.groupby("movieId")["rating"].agg(["mean", "count"])
    stats = stats[stats["count"] >= min_ratings].sort_values("mean", ascending=False).head(n)
    result = movies_df.set_index("movieId").loc[stats.index][["title", "genres"]].copy()
    result["avg_rating"] = stats["mean"].values
    result["n_ratings"] = stats["count"].values
    return result.reset_index()


if __name__ == "__main__":
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    print("=== Popularity baseline (top 5) ===")
    print(popularity_baseline(ratings, movies, n=5), "\n")

    print("=== Training SVD model ===")
    model = SVDRecommender(n_factors=20).fit(ratings)

    print("\n=== SVD recommendations for User 1 ===")
    print(model.recommend_for_user(user_id=1, movies_df=movies, n=5))
