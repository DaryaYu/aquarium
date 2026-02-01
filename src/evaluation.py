import pandas as pd
import numpy as np


def temporal_split(
        df: pd.DataFrame,
        test_ratio: float = 0.2,
        val_ratio: float = None,
        time_column_name: str = 'timestamp'
    ) -> tuple[pd.DataFrame, ...]:
    df[time_column_name] = pd.to_datetime(df[time_column_name], unit='s')
    df.sort_values(time_column_name, ascending=True, inplace=True)

    if val_ratio is None:
        split_time = df[time_column_name].quantile(1 - test_ratio)

        train = df[df[time_column_name] <= split_time]
        test = df[df[time_column_name] > split_time]

        print(f'Train set size: {train.shape}')
        print(f'Test set size: {test.shape}')
        print(f'Train timeframe: {train[time_column_name].min()} - {train[time_column_name].max()}')
        print(f'Test timeframe: {test[time_column_name].min()} - {test[time_column_name].max()}')

        return train, test

    else:
        test_split_time = df[time_column_name].quantile(1 - test_ratio)
        val_split_time = df[time_column_name].quantile(1 - test_ratio - val_ratio)

        train = df[df[time_column_name] <= val_split_time]
        val = df[(df[time_column_name] > val_split_time) & (df[time_column_name] <= test_split_time)]
        test = df[df[time_column_name] > test_split_time]

        print(f'Train set size: {train.shape}')
        print(f'Validation set size: {val.shape}')
        print(f'Test set size: {test.shape}')
        print(f'Train timeframe: {train[time_column_name].min()} - {train[time_column_name].max()}')
        print(f'Val timeframe: {val[time_column_name].min()} - {val[time_column_name].max()}')
        print(f'Test timeframe: {test[time_column_name].min()} - {test[time_column_name].max()}')

        return train, val, test


def evaluate_rmse(
        test: pd.DataFrame, 
        predict_fn,
        **predict_kwargs
) -> float:
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


def evaluate_mae(
        test: pd.DataFrame,
        predict_fn,
        **predict_kwargs
) -> float:
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

    return np.mean(np.abs(np.array(preds) - np.array(actuals)))


def evaluate_mape(
        test: pd.DataFrame,
        predict_fn,
        **predict_kwargs
) -> float:
    preds, actuals = [], []

    for row in test.itertuples():
        user_id = row.user_id
        movie_id = row.movie_id

        pred = predict_fn(
            user_id=user_id,
            movie_id=movie_id,
            **predict_kwargs
        )

        if not np.isnan(pred) and row.rating != 0:  
            preds.append(pred)
            actuals.append(row.rating)

    preds = np.array(preds)
    actuals = np.array(actuals)

    return np.mean(np.abs((actuals - preds) / actuals)) * 100


def evaluate_precision_at_k(
    test: pd.DataFrame,
    recommend_k_fn,
    k: int = 10,
    **recommend_kwargs
):

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

