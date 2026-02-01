import pandas as pd
import numpy as np


def temporal_split(
        df: pd.DataFrame, 
        test_ratio: float = 0.2, 
        time_column_name: str = 'timestamp'
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into train and test sets using a temporal (time-based) 
    strategy across all users.
    The data is first converted to datetime format and sorted chronologically.
    A cutoff time is then computed as the (1 - test_ratio) quantile of the timestamps.
    All interactions occurring at or before the split time are assigned to the training set,
    while interactions occurring after the split time are assigned to the test set.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing user-item interactions.
        Must include a column with timestamp information.
    test_ratio : float, optional (default=0.2)
        Proportion of the dataset to allocate to the test set,
        determined by the upper quantile of the timestamp distribution.
    time_column_name : str, optional (default='timestamp')
        Name of the column containing timestamp information.
        The timestamps are assumed to be in Unix seconds.

    Returns
    -------
    train : pd.DataFrame
        Training set containing data that occurred at or before the split time.
    test : pd.DataFrame
        Test set containing data that occurred after the split time.
    """
    
    df[time_column_name] = pd.to_datetime(df[time_column_name], unit='s')
    df.sort_values(time_column_name, ascending=True, inplace=True)

    split_time = df[time_column_name].quantile(1-test_ratio)
    
    train = df[df[time_column_name] <= split_time]
    test = df[df[time_column_name] > split_time]

    print(f'Train set size is: {train.shape} \nTest set size is: {test.shape}')
    print(f'Train set timeframes are: {train[time_column_name].min()} - {train[time_column_name].max()} \nTest set timeframes are {test[time_column_name].min()} - {test[time_column_name].max()}')
    
    return train, test


def evaluate_rmse(
        test: pd.DataFrame, 
        predict_fn: callable,
        **predict_kwargs
) -> float:
    """
    Evaluate rating prediction model using Root Mean Squared Error (RMSE).
    The function computes RMSE by comparing predicted ratings with true ratings
    on test set. For each (user, item) interaction in the test data,
    the provided prediction function is called to generate rating estimate.
    Predictions that return NaN are excluded from the evaluation.

    Parameters
    ----------
    test : pd.DataFrame
        Test dataset containing user-item interactions.
        Expected columns include at least:
        - 'user_id'
        - 'movie_id'
        - 'rating'
    predict_fn : callable
        Function that predicts rating for the given user and item.
        It must accept 'user_id' and 'movie_id' as keyword arguments and
        return a single float value or NaN if a prediction cannot be calculated.
    **predict_kwargs
        Additional keyword arguments passed to predict_fn. 

    Returns
    -------
    float
        Mean RMSE computed over all valid predictions.
    """
    
    preds, actuals = [], []

    for row in test.itertuples():
        user_id = row.user_id
        movie_id = row.movie_id

        pred = predict_fn(
            user_id=user_id, 
            movie_id=movie_id,
            **predict_kwargs
        )
        
        if not np.isnan(pred):
            preds.append(pred)
            actuals.append(row.rating)

    return np.sqrt(np.mean((np.array(preds) - np.array(actuals)) ** 2))


def evaluate_mape(
        test: pd.DataFrame, 
        predict_fn: callable,
        **predict_kwargs
) -> float:
    """
    Evaluate rating prediction model using Mean Absolute Percentage Error (MAPE).
    The function computes MAPE by comparing predicted ratings with true ratings
    on test set. For each (user, item) interaction in the test data,
    the provided prediction function is called to generate rating estimate.
    Predictions that return NaN are excluded from the evaluation.

    Parameters
    ----------
    test : pd.DataFrame
        Test dataset containing user-item interactions.
        Expected columns include at least:
        - 'user_id'
        - 'movie_id'
        - 'rating'
    predict_fn : callable
        Function that predicts rating for the given user and item.
        It must accept 'user_id' and 'movie_id' as keyword arguments and
        return a single float value or NaN if a prediction cannot be calculated.
    **predict_kwargs
        Additional keyword arguments passed to predict_fn. 

    Returns
    -------
    float
        MAPE computed over all valid predictions.
    """
    
    preds, actuals = [], []

    for row in test.itertuples():
        user_id = row.user_id
        movie_id = row.movie_id

        pred = predict_fn(
            user_id=user_id, 
            movie_id=movie_id,
            **predict_kwargs
        )
        
        if not np.isnan(pred):
            preds.append(pred)
            actuals.append(row.rating) 

    return np.mean(np.abs((np.array(actuals) - np.array(preds)) / np.array(actuals))) * 100


def evaluate_precision_at_k(
    test: pd.DataFrame,
    recommend_k_fn: callable,
    k: int = 10,
    **recommend_kwargs
) -> float:
    """
    Evaluate a recommender system using Precision@K metrics.
    The function measures how many of the top-K items recommended to a user
    are relevant, averaged across all users in the test set. For each user,
    a recommendation function is called to generate K recommended items.
    An item is considered relevant if its true rating in the test set is 4 or higher.

    Parameters
    ----------
    test : pd.DataFrame
        Test dataset containing user-item interactions.
        Expected columns include at least:
        - 'user_id'
        - 'movie_id'
        - 'rating'
    recommend_k_fn : callable
        Function that generates top-K recommendations for a given user.
        It must accept 'user_id' and 'k' as keyword arguments and return
        an array of recommended movie_id.
    k : int, optional (default=10)
        Number of items to recommend for user.
    **recommend_kwargs
        Additional keyword arguments passed to recommend_k_fn.

    Returns
    -------
    float
        Mean Precision@K across all users in the test set.
    """
    
    precisions = []
    for user_id in test.user_id.unique():
        rec = recommend_k_fn(
            user_id=user_id,
            test=test,
            k=k,
            **recommend_kwargs
        )
        fact = test[(test.user_id == user_id) & (test.rating>=4)].movie_id.values
        precisions.append(len(np.intersect1d(rec, fact)) / k)
        
    return sum(precisions) / len(precisions)

