import pandas as pd
import numpy as np


def temporal_split(
        df: pd.DataFrame, 
        test_ratio: float = 0.2, 
        time_column_name: str = 'timestamp'
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    
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
        train_prep: pd.DataFrame, 
        sim_df: pd.DataFrame, 
        predict_fn
        ) -> float:
    preds, actuals = [], []

    for row in test.itertuples():
        user_id = row.user_id
        movie_id = row.movie_id

        pred = predict_fn(
            user_id=user_id, 
            movie_id=movie_id, 
            train_prep=train_prep, 
            sim_df=sim_df
        )
        
        if not np.isnan(pred):
            preds.append(pred)
            actuals.append(row.rating)

    return np.sqrt(np.mean((np.array(preds) - np.array(actuals)) ** 2))

