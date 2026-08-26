"""
Evaluation for the recommender.

Two different questions matter here, so we use two different metrics:
1. "How close are predicted ratings to real ratings?" -> RMSE
2. "How many of our top-N recommendations does the user actually like?" -> Precision@K

A time-based split (train on older ratings, test on newer ones) is more
realistic than a random split, since it mimics how the system would
actually be used: predicting future preferences from past behavior.
"""
import numpy as np
import pandas as pd
from collaborative_svd import SVDRecommender, popularity_baseline


def time_based_split(ratings_df, test_frac=0.2):
    ratings_df = ratings_df.sort_values("timestamp")
    split_idx = int(len(ratings_df) * (1 - test_frac))
    train = ratings_df.iloc[:split_idx].copy()
    test = ratings_df.iloc[split_idx:].copy()
    return train, test


def rmse(model, test_df):
    errors = []
    for _, row in test_df.iterrows():
        pred = model.predict(row["userId"], row["movieId"])
        errors.append((pred - row["rating"]) ** 2)
    return np.sqrt(np.mean(errors))


def precision_at_k(model, train_df, test_df, movies_df, k=10, rating_threshold=4.0):
    """
    For each user in the test set: does the model's top-K list include
    movies the user actually rated >= threshold in the test period?
    """
    precisions = []
    test_users = test_df["userId"].unique()

    for user_id in test_users:
        user_test = test_df[test_df["userId"] == user_id]
        relevant = set(user_test[user_test["rating"] >= rating_threshold]["movieId"])
        if not relevant:
            continue

        recs = model.recommend_for_user(user_id, movies_df, n=k)
        if recs.empty:
            continue
        recommended_ids = set(recs["movieId"])

        hits = len(recommended_ids & relevant)
        precisions.append(hits / k)

    return np.mean(precisions) if precisions else 0.0


if __name__ == "__main__":
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    train, test = time_based_split(ratings, test_frac=0.2)
    print(f"Train: {len(train)} ratings | Test: {len(test)} ratings\n")

    results = {}
    for n_factors in [5, 20, 50]:
        model = SVDRecommender(n_factors=n_factors).fit(train)
        score = rmse(model, test)
        precision = precision_at_k(model, train, test, movies, k=10)
        results[n_factors] = (score, precision)
        print(f"n_factors={n_factors:3d} | RMSE={score:.4f} | Precision@10={precision:.4f}")

    print("\nNote: RMSE tends to improve then plateau/worsen as n_factors grows")
    print("(too many factors overfits noise in a small ratings matrix like this one).")
    print("On this synthetic dataset don't over-interpret absolute Precision@10 values --")
    print("re-run this same script on real MovieLens data for meaningful numbers.")
