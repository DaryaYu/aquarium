import numpy as np
import pandas as pd


def solve_with_svd(ratings_matrix, k=20, n_epochs=100, lr=0.005, reg=0.02):
    # Get dimensions
    n_users, n_items = ratings_matrix.shape

    # Initialize U and V randomly (instead of computing eigendecomposition)
    U = np.random.normal(0, 0.1, (n_users, k))
    V = np.random.normal(0, 0.1, (n_items, k))

    # Get mask of observed ratings (non-zero/non-NaN entries)
    mask = ~np.isnan(ratings_matrix.values) & (ratings_matrix.values != 0)
    observed_indices = np.argwhere(mask)

    # Gradient descent on observed entries only
    for epoch in range(n_epochs):
        for user_idx, item_idx in observed_indices:
            # Predict rating
            prediction = np.dot(U[user_idx], V[item_idx])

            # Compute error
            actual_rating = ratings_matrix.iloc[user_idx, item_idx]
            error = actual_rating - prediction

            # Update factors (gradient descent with regularization)
            U[user_idx] += lr * (error * V[item_idx] - reg * U[user_idx])
            V[item_idx] += lr * (error * U[user_idx] - reg * V[item_idx])

    # Reconstruct full matrix
    approx_matrix = np.dot(U, V.T)

    return pd.DataFrame(
        approx_matrix, index=ratings_matrix.index, columns=ratings_matrix.columns
    )
