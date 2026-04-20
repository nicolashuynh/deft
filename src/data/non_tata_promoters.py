from typing import Sequence, Union

import numpy as np
import pandas as pd
from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanNontataPromoters


def extract_X_y(dset):
    """
    Extracts the features and labels from the dataset.
    """
    length = len(dset)
    # Iterate through the dataset and extract the features and labels
    X = []
    y = []
    for i in range(length):
        # Get the features and labels
        x, y_i = dset[i]
        # Append the features and labels to the lists
        X.append(x)
        y.append(y_i)
    # Convert the lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y


def dna_sequences_to_dataframe(
    dna_array: Union[Sequence[str], np.ndarray], use_raw_sequence: bool = False
) -> pd.DataFrame:
    """Handle dna sequences to dataframe."""
    is_empty = False
    if dna_array is None:
        is_empty = True
    # Check if it has a 'size' attribute (like NumPy arrays) and if it's 0
    elif hasattr(dna_array, "size") and dna_array.size == 0:
        is_empty = True
    # Fallback for standard sequences (like lists) without 'size'
    elif not hasattr(dna_array, "size") and len(dna_array) == 0:
        is_empty = True

    if is_empty:
        # Return an empty DataFrame for empty input
        return pd.DataFrame()

    if not use_raw_sequence:
        list_of_chars = [list(seq) for seq in dna_array]

        df = pd.DataFrame(list_of_chars)

        max_pos = df.shape[1]

        column_names = [f"seq_{i}" for i in range(max_pos)]

        df.columns = column_names
    else:
        df = pd.DataFrame(dna_array, columns=["raw_sequence"])

    return df


def load_promoters(use_raw_sequence=False):
    """Load promoters."""
    dset_train = HumanNontataPromoters(split="train", version=0)
    dset_test = HumanNontataPromoters(split="test", version=0)
    X_train, y_train = extract_X_y(dset_train)
    X_test, y_test = extract_X_y(dset_test)

    df_train = dna_sequences_to_dataframe(X_train, use_raw_sequence)
    df_test = dna_sequences_to_dataframe(X_test, use_raw_sequence)

    n_train = len(df_train)

    df_whole = pd.concat([df_train, df_test], axis=0)

    if not use_raw_sequence:
        # One hot encode
        df_whole = pd.get_dummies(df_whole)

    # Split back into train and test
    df_train = df_whole.iloc[:n_train, :]
    df_test = df_whole.iloc[n_train:, :]

    return df_train, df_test, y_train, y_test
