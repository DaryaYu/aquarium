import os
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_DATA_FOLDER_PATH = ROOT / 'data' / 'ml-1m'


def read_file(file_name: str, columns: list): 
    file_path = RAW_DATA_FOLDER_PATH / file_name
    return pd.read_csv(
        file_path,
        sep='::',
        engine='python',
        encoding='latin-1',
        names=columns
    )


def read_movies_file() -> pd.DataFrame:
    return read_file(
        file_name='movies.dat',
        columns=['movie_id', 'title', 'genres']
    )


def read_users_file() -> pd.DataFrame:
    return read_file(
        file_name='users.dat',
        columns=['user_id', 'gender', 'age', 'occupation', 'zip']
    )


def read_ratings_file() -> pd.DataFrame:
    return read_file(
        file_name='ratings.dat',
        columns=['user_id', 'movie_id', 'rating', 'timestamp']
    )



