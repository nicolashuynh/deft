from pathlib import Path

import h5py
import numpy as np


def load_mpra(condition_name, one_hot_encode=True, use_2D=False):
    """
    Loads and preprocesses the MPRA dataset.
    """
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "mpra" / "processed_enhancers"

    print(
        f"--- Loading data for '{condition_name}' with one_hot_encode={one_hot_encode} ---"
    )

    train_file = data_dir / f"train_balanced_{condition_name}.hdf5"
    test_file = data_dir / f"test_balanced_{condition_name}.hdf5"

    def _load_h5_file(file_path):
        """Load h5 file."""
        with h5py.File(file_path, "r") as hf:
            return hf["X"][:], hf["Y"][:]

    # Load the data
    X_train, y_train = _load_h5_file(train_file)
    X_test, y_test = _load_h5_file(test_file)

    if one_hot_encode:
        if use_2D:
            # Reshape the 3D array (samples, length, 4) into a 2D array (samples, features)
            num_samples_train, seq_len, alphabet_size = X_train.shape
            X_train = X_train.reshape((num_samples_train, seq_len * alphabet_size))

            num_samples_test, _, _ = X_test.shape
            X_test = X_test.reshape((num_samples_test, seq_len * alphabet_size))

            print("Returned X data as flattened one-hot arrays.")
        else:
            print("Keep the 3D shape of the array")

    else:

        def _one_hot_to_strings(one_hot_array):
            # Mapping from index to nucleotide character
            """Handle one hot to strings."""
            idx_to_nuc = {0: "A", 1: "C", 2: "G", 3: "T"}
            indices = np.argmax(one_hot_array, axis=2)
            nuc_array = np.vectorize(idx_to_nuc.get)(indices)
            return np.array(["".join(row) for row in nuc_array])

        X_train = _one_hot_to_strings(X_train)
        X_test = _one_hot_to_strings(X_test)

        print("Returned X data as an array of DNA sequence strings.")

    return X_train, y_train, X_test, y_test
