import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


@dataclass
class FeatureInfo:
    fn: Callable = None
    name: str = None
    description: str = None
    string: str = None
    threshold: float = None
    score: float = None
    rationale: str = None
    

    def __str__(self):
        """Return the string representation."""
        base_str = f"""
    Name: {self.name}
    Description: {self.description}
    Rationale: {self.rationale}
    Code: {self.string}
    """
        return base_str

    def __eq__(self, value) -> bool:
        """Handle eq."""
        return self.score == value.score

    def __hash__(self) -> int:
        """Handle hash."""
        return hash(self.score)


def list_to_str(list_features):
    """Handle list to str."""
    return "\n".join(f"Feature {i+1}:{str(obj)}" for i, obj in enumerate(list_features))


def get_naive_features(X):
    """
    Generate naive features for sequence analysis.
    """
    list_features = []

    use_raw_sequence = "raw_sequence" in X.columns

    if use_raw_sequence:
        # Get length from first sequence
        seq_length = len(X["raw_sequence"].iloc[0])
        

        list_pos = [i for i in range(seq_length)]
        assert (
            len(list_pos) == seq_length
        ), "Length of list_pos must match sequence length"

        # Get the nucleotides as the list of unique characters in the raw sequence
        nucleotides = list(set("".join(X["raw_sequence"].iloc[0])))
    else:
        # Count sequence position columns (seq_0, seq_1, etc.)
        seq_length = sum(1 for col in X.columns if col.startswith("seq_"))

        list_pos = [i for i in range((-seq_length // 2) + 1, seq_length // 2 + 1)]

        logging.info(list_pos)
        assert (
            len(list_pos) == seq_length
        ), "Length of list_pos must match sequence length"

        nucleotides = ["A", "C", "G", "T"]

    # Generate features for each position and nucleotide
    for pos in list_pos:
        pos_str = f"neg{abs(pos)}" if pos < 0 else str(pos)

        for nuc in nucleotides:
            if use_raw_sequence:
                # Create function for raw sequence analysis
                func_str = f"""def pos_{pos_str}_is_{nuc}(X):
    return X['raw_sequence'].apply(lambda x: x[{pos}] == '{nuc}')"""

                feature_desc = f"Check if position {pos} in raw sequence is {nuc}"

            else:
                # Create function for positional column analysis
                func_str = f"""def pos_{pos_str}_is_{nuc}(X):
    return X['seq_{pos}'] == '{nuc}'"""

                feature_desc = f"Check if nucleotide at position {pos} is {nuc}"

            # Create FeatureInfo object for this feature
            feature = FeatureInfo(
                name=f"pos_{pos}_is_{nuc}",
                description=feature_desc,
                rationale=f"Basic sequence composition feature checking for {nuc} at position {pos}",
                string=func_str,
            )

            # Convert the function string to callable
            namespace = {}
            exec(func_str, namespace)
            feature.fn = namespace[f"pos_{pos_str}_is_{nuc}"]

            list_features.append(feature)

    return list_features
