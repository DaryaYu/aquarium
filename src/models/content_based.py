import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def create_item_representation(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Create explicit item representation using TF-IDF on movie genres.
    
    Item Representation:
    - Each movie is represented as a TF-IDF vector over its genres
    - Genres are treated as a "document" (e.g., "Action|Adventure|Sci-Fi")
    - TF-IDF captures the importance of each genre for each movie
    - This creates a sparse, high-dimensional feature space
    
    Justification:
    - TF-IDF is appropriate for categorical text features (genres)
    - It normalizes for genre frequency (common genres get lower weight)
    - Creates a dense representation suitable for similarity computation
    
    Parameters
    ----------
    movies : pd.DataFrame
        DataFrame with columns ['movie_id', 'title', 'genres']
        Genres are pipe-separated strings (e.g., "Action|Adventure|Sci-Fi")
    
    Returns
    -------
    pd.DataFrame
        Item representation matrix with movies as rows and genre features as columns
    """
    genres_text = movies['genres'].fillna('').str.replace('|', ' ')
    vectorizer = TfidfVectorizer(min_df=1, max_features=None, token_pattern=r'\S+')
    item_representation = vectorizer.fit_transform(genres_text)
    
    feature_names = vectorizer.get_feature_names_out()
    item_df = pd.DataFrame(
        item_representation.toarray(),
        index=movies['movie_id'],
        columns=feature_names
    )
    
    return item_df


def compute_item_similarity(item_representation: pd.DataFrame, similarity_type: str = 'cosine') -> pd.DataFrame:
    """
    Compute item-item similarity matrix using specified similarity function.
    
    Parameters
    ----------
    item_representation : pd.DataFrame
        Item representation matrix (movies x genres)
    similarity_type : str, optional (default='cosine')
        Type of similarity to compute: 'cosine' or 'euclidean'
    
    Returns
    -------
    pd.DataFrame
        Item-item similarity matrix (movies x movies)
    """
    if similarity_type == 'cosine':
        similarity_matrix = cosine_similarity(item_representation.values)
    elif similarity_type == 'euclidean':
        distance_matrix = euclidean_distances(item_representation.values)
        similarity_matrix = 1 / (1 + distance_matrix)
    else:
        raise ValueError(f"Unknown similarity_type: {similarity_type}. Use 'cosine' or 'euclidean'.")
    
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=item_representation.index,
        columns=item_representation.index
    )
    
    return similarity_df


def predict_rating_content_based(
    user_id: int,
    movie_id: int,
    train_prep: pd.DataFrame,
    sim_df: pd.DataFrame,
    n: int = 10
) -> float:
    """
    Predict a user rating for a movie using content-based filtering.
    
    The prediction is based on:
    1. User's historical ratings for movies
    2. Content similarity between the target movie and rated movies
    3. Weighted average of ratings, weighted by similarity
    
    Parameters
    ----------
    user_id : int
        ID of the user
    movie_id : int
        ID of the target movie
    train_prep : pd.DataFrame
        User-item rating matrix used for training
    sim_df : pd.DataFrame
        Item-item similarity matrix (content-based)
    n : int, optional (default=10)
        Number of most similar items to use for prediction
    
    Returns
    -------
    float
        Predicted rating or NaN if user_id or movie_id is missing
    """
    if user_id not in train_prep.index or movie_id not in train_prep.columns:
        return np.nan
    
    user_ratings = train_prep.loc[user_id]
    valid_items_list = user_ratings.drop(movie_id).dropna().index
    
    if len(valid_items_list) == 0:
        return np.nan
    
    top_n_similarity = sim_df[movie_id][valid_items_list]\
        .sort_values(ascending=False)\
        .head(n)
    
    if len(top_n_similarity) == 0 or top_n_similarity.sum() == 0:
        return np.nan
    
    top_n_ratings = user_ratings.loc[top_n_similarity.index]
    prediction = np.dot(top_n_similarity.values, top_n_ratings.values) / np.sum(top_n_similarity.values)
    
    return prediction
