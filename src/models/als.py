import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve


def solve_with_als(
    ratings_matrix,
    alpha=40,
    iterations=10,
    factors=20,
    regularization=0.1,
    random_seed=42,
    verbose=True,
):
    if verbose:
        print(f"--- Running ALS with parameters ---")
        print(f"Alpha: {alpha}")
        print(f"Iterations: {iterations}")
        print(f"Factors: {factors}")
        print(f"Regularization: {regularization}")

    confidence_matrix = alpha * ratings_matrix
    num_users, num_items = confidence_matrix.shape

    random_state = np.random.RandomState(random_seed)
    user_matrix = sparse.csr_matrix(random_state.normal(size=(num_users, factors)))
    item_matrix = sparse.csr_matrix(random_state.normal(size=(num_items, factors)))

    user_eye = sparse.eye(num_users)
    item_eye = sparse.eye(num_items)
    lambda_eye = regularization * sparse.eye(factors)

    iterator = range(iterations) if verbose else range(iterations)
    for iter_num in iterator:
        if verbose:
            print(f"\nIteration {iter_num + 1}/{iterations}")

        userTuser = user_matrix.T.dot(user_matrix)
        itemTitem = item_matrix.T.dot(item_matrix)

        if verbose:
            print("Solving for users (fixed items)...")
        for user in range(num_users):
            confidence_row = confidence_matrix.iloc[user].to_numpy().copy()
            preference_vector = confidence_row.copy()
            preference_vector[preference_vector != 0] = 1
            confidence_row += 1
            CuI = sparse.diags(confidence_row, 0)
            itemTCuIitem = item_matrix.T.dot(CuI).dot(item_matrix)
            itemTCuPu = item_matrix.T.dot(CuI + item_eye).dot(preference_vector.T)
            user_matrix[user] = spsolve(
                itemTitem + itemTCuIitem + lambda_eye, itemTCuPu
            )

        if verbose:
            print("Solving for items (fixed users)...")
        for item in range(num_items):
            confidence_row = confidence_matrix.T.iloc[item].to_numpy().copy()
            preference_vector = confidence_row.copy()
            preference_vector[preference_vector != 0] = 1
            confidence_row += 1
            CiI = sparse.diags(confidence_row, 0)
            userTCiIuser = user_matrix.T.dot(CiI).dot(user_matrix)
            userTCiPi = user_matrix.T.dot(CiI + user_eye).dot(preference_vector.T)
            item_matrix[item] = spsolve(
                userTuser + userTCiIuser + lambda_eye, userTCiPi
            )

    predictions = user_matrix.dot(item_matrix.T).toarray()

    return pd.DataFrame(
        predictions, index=ratings_matrix.index, columns=ratings_matrix.columns
    )
