from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from src.heuristic_ranking import (
    popularity_based_ranking,
    recency_based_ranking,
    pagerank_ranking,
)


def _minmax_normalize(series: pd.Series) -> pd.Series:

    if series.empty:
        return series
    min_val = series.min()
    max_val = series.max()
    if np.isclose(max_val, min_val):
        return pd.Series(np.zeros_like(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def item_based_scores(
    interactions: pd.DataFrame,
    user_column: str = "user_id",
    item_column: str = "movie_id",
    rating_column: str = "rating",
    filter_seen: bool = True,
) -> pd.DataFrame:
    user_item_matrix = interactions.pivot_table(
        index=user_column,
        columns=item_column,
        values=rating_column,
        fill_value=0.0,
    )

    item_vectors = user_item_matrix.T  
    similarity = cosine_similarity(item_vectors)

    numerator = user_item_matrix.values.dot(similarity)
    denom = np.abs(similarity).sum(axis=1)
    denom = np.where(denom == 0, 1e-9, denom)
    pred = numerator / denom

    pred_df = pd.DataFrame(
        pred,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns,
    )

    result = (
        pred_df.stack()
        .reset_index()
        .rename(columns={"level_0": user_column, "level_1": item_column, 0: "collab_score"})
    )

    if filter_seen:
        seen = interactions.groupby(user_column)[item_column].apply(set).to_dict()
        mask = result.apply(
            lambda row: row[item_column] not in seen.get(row[user_column], set()), axis=1
        )
        result = result[mask]

    return result[[user_column, item_column, "collab_score"]]


def compute_heuristic_scores(
    interactions: pd.DataFrame,
    item_column: str = "movie_id",
    method: str = "popularity",
    user_column: str = "user_id",
) -> pd.Series:
    
    if method == "popularity":
        rating_column = "rating" if "rating" in interactions.columns else None
        scores = popularity_based_ranking(
            interactions,
            item_column=item_column,
            rating_column=rating_column,
            weight_by_mean_rating=True,
            k=None,
        )
    elif method == "recency":
        timestamp_column = "timestamp" if "timestamp" in interactions.columns else None
        if timestamp_column is None:
            raise ValueError("recency heuristic requires a timestamp column")
        scores = recency_based_ranking(
            interactions,
            item_column=item_column,
            timestamp_column=timestamp_column,
            k=None,
        )
    elif method == "pagerank":
        scores = pagerank_ranking(
            interactions,
            user_column=user_column,
            item_column=item_column,
        )
    else:
        raise ValueError(f"Unsupported heuristic method: {method}")
    return scores


def weighted_hybrid_recommendation(
    interactions: pd.DataFrame,
    alpha: float = 0.5,
    heuristics: str = "popularity",
    k: int = 10,
    user_column: str = "user_id",
    item_column: str = "movie_id",
    rating_column: str = "rating",
    user_subset: Optional[Sequence] = None,
) -> pd.DataFrame:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    collab_df = item_based_scores(
        interactions,
        user_column=user_column,
        item_column=item_column,
        rating_column=rating_column,
        filter_seen=True,
    )

    if user_subset is not None:
        collab_df = collab_df[collab_df[user_column].isin(user_subset)]

    collab_df["collab_norm"] = _minmax_normalize(collab_df["collab_score"])
    heuristic_scores = compute_heuristic_scores(
        interactions,
        item_column=item_column,
        method=heuristics,
        user_column=user_column,
    )
    heuristic_scores_norm = _minmax_normalize(heuristic_scores)
    heuristic_dict = heuristic_scores_norm.to_dict()

    collab_df["heuristic_norm"] = collab_df[item_column].map(heuristic_dict).fillna(0.0)

    collab_df["hybrid_score"] = alpha * collab_df["heuristic_norm"] + (1.0 - alpha) * collab_df["collab_norm"]

    collab_df.sort_values(by=[user_column, "hybrid_score"], ascending=[True, False], inplace=True)

    topk_df = collab_df.groupby(user_column).head(k).reset_index(drop=True)
    return topk_df[[user_column, item_column, "hybrid_score"]]