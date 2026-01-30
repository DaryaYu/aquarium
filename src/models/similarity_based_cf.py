import numpy as np
import pandas as pd


def predict_rating_cf_item_based(
        user_id: int, 
        movie_id: int, 
        train_prep: pd.DataFrame,
        sim_df: pd.DataFrame,
        n: int = 10
    ) -> float:

    if user_id not in train_prep.index or movie_id not in train_prep.columns:
       # print(f'Either user id {user_id} or movie id {movie_id} is missing in the data sample')
        return np.nan

    # Get the user's movie ratings 
    user_ratings = train_prep.loc[user_id]

    # Get the list of movies with valid ratings
    valid_items_list = user_ratings.drop(movie_id).dropna().index

    # Get top k movies by similarity
    top_n_similarity = sim_df[movie_id][valid_items_list]\
        .sort_values(ascending=False)\
        .head(n)

    # Get the user ratings for the top k movies
    top_n_ratings = user_ratings.loc[top_n_similarity.index]

    return np.dot(top_n_similarity.values, top_n_ratings.values) / np.sum(top_n_similarity.values)


def predict_rating_cf_user_based(
        user_id: int, 
        movie_id: int, 
        train_prep: pd.DataFrame,
        sim_df: pd.DataFrame,
        n: int = 10
    ) -> float:

    if user_id not in train_prep.index or movie_id not in train_prep.columns:
       # print(f'Either user id {user_id} or movie id {movie_id} is missing in the data sample')
        return np.nan

    # Get all users who rated the movie
    users_rated = train_prep[movie_id].dropna()

    # Get top k neighbors that rated the movie
    neighbors = sim_df.loc[user_id]\
        .drop(user_id)[users_rated.index]\
        .sort_values(ascending=False)\
        .head(n)

    # Get ratings of top k neighbors
    neighbor_ratings = users_rated.loc[neighbors.index]

    return np.dot(neighbors.values, neighbor_ratings.values) / np.sum(neighbors.values)


def recommend(
    user_id: int,
    test: pd.DataFrame,
    predict_fn,
    train_prep: pd.DataFrame,
    sim_df: pd.DataFrame,
    n: int, 
    k: int = 10
):            

    movie_list = test.movie_id.unique()
    rated_movies = train_prep.loc[user_id].dropna().index.values
    candidates = np.setdiff1d(movie_list, rated_movies)
    
    pred_ratings = []
    for movie_id in candidates:
        pred_rating = predict_fn(
            user_id=user_id, 
            movie_id=movie_id, 
            train_prep=train_prep, 
            sim_df=sim_df,
            n=n
        )
        pred_ratings.append(pred_rating)
    
    return pd.DataFrame(pred_ratings, index=candidates).sort_values(by=0, ascending=False).head(k).index.values
    
