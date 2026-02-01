import numpy as np
import pandas as pd


def solve_with_svd(
    ratings_matrix, k=20, n_epochs=100, lr=0.005, reg=0.02, verbose=True
):

    n_users, n_items = ratings_matrix.shape

    U = np.random.normal(0, 0.1, (n_users, k))
    V = np.random.normal(0, 0.1, (n_items, k))

    mask = ~np.isnan(ratings_matrix.values) & (ratings_matrix.values != 0)
    observed_indices = np.argwhere(mask)
    n_observed = len(observed_indices)

    rmse_history = []

    epoch_iterator = range(n_epochs)

    for epoch in epoch_iterator:
        if verbose and epoch%5 == 0: 
            print(f"Epoch {epoch + 1}/{n_epochs}")
        np.random.shuffle(observed_indices)

        epoch_error = 0.0
        for user_idx, item_idx in observed_indices:
            prediction = np.dot(U[user_idx], V[item_idx])

            actual_rating = ratings_matrix.iloc[user_idx, item_idx]
            error = actual_rating - prediction
            epoch_error += error**2

            U[user_idx] += lr * (error * V[item_idx] - reg * U[user_idx])
            V[item_idx] += lr * (error * U[user_idx] - reg * V[item_idx])

        rmse = np.sqrt(epoch_error / n_observed)
        rmse_history.append(rmse)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}: RMSE = {rmse:.4f}")

    approx_matrix = np.dot(U, V.T)

    return pd.DataFrame(
        approx_matrix, index=ratings_matrix.index, columns=ratings_matrix.columns
    )
