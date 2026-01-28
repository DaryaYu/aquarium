import numpy as np
import pandas as pd


def predict_rating_cf_item_based(
        user_id: int, 
        movie_id: int, 
        train_prep: pd.DataFrame,
        sim_df: pd.DataFrame,
        k: int = 10
    ) -> float:

    if user_id not in train_prep.index or movie_id not in train_prep.columns:
       # print(f'Either user id {user_id} or movie id {movie_id} is missing in the data sample')
        return np.nan

    # Get the user's movie ratings 
    user_ratings = train_prep.loc[user_id]

    # Get the list of movies with valid ratings
    valid_items_list = user_ratings.drop(movie_id).dropna().index

    # Get top k movies by similarity
    top_k_similarity = sim_df[movie_id][valid_items_list]\
        .sort_values(ascending=False).head(k)

    # Get the user ratings for the top k movies
    top_k_ratings = user_ratings.loc[top_k_similarity.index]

    return np.dot(top_k_similarity.values, top_k_ratings.values) / np.sum(top_k_similarity.values)


def predict_rating_cf_user_based(
        user_id: int, 
        movie_id: int, 
        train_prep: pd.DataFrame,
        sim_df: pd.DataFrame,
        k: int = 10
    ) -> float:

    if user_id not in train_prep.index or movie_id not in train_prep.columns:
       # print(f'Either user id {user_id} or movie id {movie_id} is missing in the data sample')
        return np.nan

    # Get all users who rated the movie
    users_rated = train_prep[movie_id].dropna()

    # Get top k neighbors that rated the movie
    neighbors = sim_df.loc[user_id]\
        .drop(user_id)[users_rated.index]\
        .sort_values(ascending=False).head(k)

    # Get ratings of top k neighbors
    neighbor_ratings = users_rated.loc[neighbors.index]

    return np.dot(neighbors.values, neighbor_ratings.values) / np.sum(neighbors.values)