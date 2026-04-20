from itertools import product

import numpy as np
import pandas as pd


def analyze_column_types(df):
    """Handle analyze column types."""
    column_types = {}

    for column in df.columns:
        dtype = df[column].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            unique_ratio = len(df[column].unique()) / len(df)
            if unique_ratio < 0.05:  
                column_types[column] = "categorical"
            else:
                if pd.api.types.is_integer_dtype(dtype):
                    column_types[column] = "discrete"
                else:
                    column_types[column] = "continuous"

        elif pd.api.types.is_datetime64_dtype(dtype):
            column_types[column] = "datetime"

        elif pd.api.types.is_bool_dtype(dtype):
            column_types[column] = "boolean"

        else:  
            unique_ratio = len(df[column].unique()) / len(df)
            if unique_ratio < 0.05:
                column_types[column] = "categorical"
            else:
                column_types[column] = "text"

    return column_types


def extract_kmer_features(
    df: pd.DataFrame, k: int, col_name: str = "raw_sequence", valid_chars: str = "ACGT"
) -> (np.ndarray, list):

    """Extract kmer features."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in DataFrame")

    n_samples = len(df)

    possible_kmers = ["".join(p) for p in product(valid_chars, repeat=k)]
    n_features = len(possible_kmers)

    kmer_to_index = {kmer: idx for idx, kmer in enumerate(possible_kmers)}

    feature_matrix = np.zeros((n_samples, n_features), dtype=int)

    for i, sequence in enumerate(df[col_name]):
        if not isinstance(sequence, str):
            print(
                f"Warning: Row {df.index[i]} in column '{col_name}' is not a string. Skipping."
            )
            continue  

        seq_len = len(sequence)
        if seq_len < k:
            continue

        for j in range(seq_len - k + 1):
            kmer = sequence[
                j : j + k
            ].upper()  

            if kmer in kmer_to_index:
                feature_index = kmer_to_index[kmer]
                feature_matrix[i, feature_index] += 1

    return feature_matrix, possible_kmers


def extract_X_y(dset):
    """
    Extracts the features and labels from the dataset.
    """
    length = len(dset)
    X = []
    y = []
    for i in range(length):
        x, y_i = dset[i]
        X.append(x)
        y.append(y_i)
    X = np.array(X)
    y = np.array(y)
    return X, y
