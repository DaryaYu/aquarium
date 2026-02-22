from __future__ import annotations

from typing import Iterable, Sequence, Optional

import numpy as np
import pandas as pd


def popularity_based_ranking(
    interactions: pd.DataFrame,
    item_column: str = "movie_id",
    rating_column: Optional[str] = "rating",
    k: Optional[int] = None,
    weight_by_mean_rating: bool = True,
) -> pd.Series:
    if interactions.empty:
        return pd.Series(dtype=float)

    counts = interactions.groupby(item_column).size().rename("count")

    if weight_by_mean_rating and rating_column is not None:
        mean_ratings = interactions.groupby(item_column)[rating_column].mean()
        popularity_scores = counts.multiply(mean_ratings, fill_value=0)
    else:
        popularity_scores = counts

    ranked = popularity_scores.sort_values(ascending=False)

    if k is not None:
        ranked = ranked.head(k)
    return ranked


def recency_based_ranking(
    interactions: pd.DataFrame,
    item_column: str = "movie_id",
    timestamp_column: str = "timestamp",
    k: Optional[int] = None,
) -> pd.Series:
    if interactions.empty:
        return pd.Series(dtype=float)

    timestamps = interactions[timestamp_column]
    if not np.issubdtype(timestamps.dtype, np.number):
        timestamps = pd.to_datetime(timestamps).astype("int64") // 10**9

    recent_times = interactions.groupby(item_column)[timestamp_column].max()
    if not np.issubdtype(recent_times.dtype, np.number):
        recent_times = pd.to_datetime(recent_times).astype('int64') // 10**9
    return recent_times.sort_values(ascending=False)

    ranked = recent_times.sort_values(ascending=False)

    if k is not None:
        ranked = ranked.head(k)
    return ranked


def pagerank_ranking(
    interactions: pd.DataFrame,
    user_column: str = "user_id",
    item_column: str = "movie_id",
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    k: Optional[int] = None,
) -> pd.Series:
    if interactions.empty:
        return pd.Series(dtype=float)

    items = interactions[item_column].unique()
    n = len(items)
    item_to_idx = {item: idx for idx, item in enumerate(items)}
    A = np.zeros((n, n), dtype=float)

    for _, group in interactions.groupby(user_column)[item_column]:
        unique_items = pd.unique(group)
        indices = [item_to_idx[i] for i in unique_items]
        for i in indices:
            for j in indices:
                if i != j:
                    A[i, j] += 1.0

    row_sums = A.sum(axis=1)
    P = np.zeros_like(A)
    for i in range(n):
        if row_sums[i] > 0:
            P[i, :] = A[i, :] / row_sums[i]
        else:
            P[i, :] = 1.0 / n

    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        r_new = alpha * P.T.dot(r) + (1.0 - alpha) / n
        if np.sum(np.abs(r_new - r)) < tol:
            r = r_new
            break
        r = r_new

    scores = pd.Series({item: r[item_to_idx[item]] for item in items})
    ranked = scores.sort_values(ascending=False)
    if k is not None:
        ranked = ranked.head(k)
    return ranked


__all__ = [
    "popularity_based_ranking",
    "recency_based_ranking",
    "pagerank_ranking",
]