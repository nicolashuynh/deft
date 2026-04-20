import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_polymerase(
    use_raw_sequence,
    type_dataset,
    no_prior_knowledge,
    test_size,
    random_state,
    size_left_window=50,
    one_hot_encoding=False,
    use_external_test_set=True,
):


    """Load polymerase."""
    if type_dataset == "Plus":
        name_dataset = (
            "pauses_combined_processed_with_signals_subsampled_Plus_Spt5_reformatted"
        )
    else:
        raise ValueError(f"Dataset type {type_dataset} not recognized")
    logging.info(f"Dataset: {name_dataset}")
 
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data"
    path_file = data_root / "polymerase" / "processed" / f"{name_dataset}.csv"

    df_rna = pd.read_csv(path_file)

    if use_raw_sequence:
        columns_X = ["raw_sequence"]
        columns_y = ["paused"]
        X = df_rna[columns_X]
        y = df_rna[columns_y]
    else:
        columns_X = [f"seq_{i}" for i in range(-size_left_window, size_left_window + 1)]
        columns_y = ["paused"]
        X = df_rna[columns_X]
        y = df_rna[columns_y]

    if no_prior_knowledge:
        # Rename the nucleotides as A to D, G to E, T to F and C to G to remove prior information
        # Define the mapping dictionary
        mapping = {"A": "D", "T": "E", "G": "F", "C": "G", "N": "N"}

        # Apply the transformation to the column
        X["raw_sequence"] = X["raw_sequence"].apply(
            lambda x: "".join(mapping[base] for base in x)
        )

    # Split into a training and test set
    if one_hot_encoding:
        # One-hot encoding of the sequences
        X = pd.get_dummies(X)

    if use_external_test_set:
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Use pre-generated external test set.
        test_file = (
            data_root
            / "polymerase"
            / "test"
            / f"test_{type_dataset.lower()}_reformatted.csv"
        )
        df_test = pd.read_csv(test_file)
        df_test = df_test.sample(frac=test_size, random_state=42).reset_index(drop=True)

        if use_raw_sequence:
            columns_X = ["raw_sequence"]
            columns_y = ["paused"]
            X_test = df_test[columns_X]
            y_test = df_test[columns_y]
        else:
            columns_X = [
                f"seq_{i}" for i in range(-size_left_window, size_left_window + 1)
            ]
            columns_y = ["paused"]
            X_test = df_test[columns_X]
            y_test = df_test[columns_y]
        if one_hot_encoding:
            X_test = pd.get_dummies(X_test)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    return X_train, X_test, y_train, y_test
