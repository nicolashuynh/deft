# Interpretable DNA Sequence Classification via Dynamic Feature Generation in Decision Trees (AISTATS 2026)

Code for our AISTATS 2026 paper
**[Interpretable DNA Sequence Classification via Dynamic Feature Generation in Decision Trees](https://arxiv.org/abs/2604.12060)** which introduces a new method called DEFT.

---

## Quick Start

### 1) Install environment

We recommend `uv` for environment management.

```bash
cd <repo-root>

# Optional: ensure Python 3.10 is available
uv python install 3.10

# Create/sync project environment from pyproject.toml
uv sync
```

Run commands with:

```bash
uv run <command>
```

### 2) Configure API keys

Keys are read from environment variables in config files.

```bash
# Generic default llm_models config (conf/llm_models/generic_user.yaml)
export DEFT_API_KEY="<your-key>"

# OpenAI API config (conf/llm_models/openai_gpt4o_mini.yaml)
export OPENAI_API_KEY="<your-key>"
```

If these variables are missing, LLM-backed runs fail at config/runtime resolution.

### 3) Run DEFT (example)

```bash
uv run python experiments/deft_tree.py --config-name=config
```

Outputs are written to Hydra run directories:

```text
outputs/<date>/<time>/
```

---

## Repository structure

- `experiments/`: DEFT and baseline experiment entry points.
- `conf/`: Hydra configs for datasets, LLM backends, tree settings, and experiments.
- `src/data/`: dataset plugins and loaders.
- `src/trees/`: tree growth and split logic.
- `src/utils/`: utilities (selection, parsing, etc.).
- `notebooks/`: reproducible analysis and paper tables/figures.
- `artifacts/trees/`: lightweight tree artifacts used by notebooks.


---

## Use DEFT on Your Own Dataset

Here is our recommended integration path:

### Step 1) Prepare data files

Example layout:

```text
data/my_dataset/processed/train.csv
data/my_dataset/processed/test.csv
```

Expected columns:

- one sequence column (for example `raw_sequence`)
- one binary label column (for example `label`)

Sequences should remain plain strings.

### Step 2) Add a dataset plugin

Create a class (for example `src/data/my_dataset_plugin.py`) inheriting `BaseDatasetPlugin`.

Required methods:

- `load_for_deft(self) -> dict`
- `load_for_baseline(self, *, featurizer_name: str, seed: int)`
- `load_for_dl(self, *, seed: int)`

Return contract:

- `load_for_deft` returns:
  - `X_train`
  - `y_train`
  - `tree_filename` (for example `tree_my_dataset.dill`)
  - `prompt_context` (use `{}` if not needed)
- `load_for_baseline` returns `(X_train, X_test, y_train, y_test)`
  - raw sequence features for `featurizer_name == "kmer_count"`
  - numeric 2D features for other featurizers
- `load_for_dl` returns `(X_train, X_test, y_train, y_test)` with numeric inputs compatible with:
  - `(-1, dataset.sequence_length, dataset.n_nucleotides)`

Minimal skeleton:

```python
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.dataset_plugins import BaseDatasetPlugin


class MyDatasetPlugin(BaseDatasetPlugin):
    dl_subsample_before_val_split = False

    def __init__(
        self,
        data_dir: str = "data/my_dataset/processed",
        train_filename: str = "train.csv",
        test_filename: str = "test.csv",
        sequence_column: str = "raw_sequence",
        label_column: str = "label",
        random_state: int = 42,
        subsample_proportion: float = 1.0,
    ):
        repo_root = Path(__file__).resolve().parents[2]
        self.data_dir = (repo_root / data_dir).resolve()
        self.train_path = self.data_dir / train_filename
        self.test_path = self.data_dir / test_filename
        self.sequence_column = sequence_column
        self.label_column = label_column
        self.random_state = int(random_state)
        self.subsample_proportion = float(subsample_proportion)

    def _load_split(self, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(path)
        X = pd.DataFrame(df[self.sequence_column].astype(str), columns=["raw_sequence"])
        y = pd.DataFrame(df[self.label_column].astype(int), columns=[self.label_column])
        return X, y

    def _load_train_test(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X_train, y_train = self._load_split(self.train_path)
        X_test, y_test = self._load_split(self.test_path)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def _one_hot_flatten(raw_sequences: pd.Series) -> pd.DataFrame:
        seq_df = pd.DataFrame(raw_sequences.apply(list).tolist())
        seq_df.columns = [f"seq_{i}" for i in range(seq_df.shape[1])]
        return pd.get_dummies(seq_df)

    def load_for_deft(self) -> dict[str, Any]:
        X_train, _, y_train, _ = self._load_train_test()
        return {
            "X_train": X_train,
            "y_train": y_train,
            "tree_filename": "tree_my_dataset.dill",
            "prompt_context": {},
        }

    def load_for_baseline(
        self, *, featurizer_name: str, seed: int
    ) -> tuple[Any, Any, Any, Any]:
        X_train, X_test, y_train, y_test = self._load_train_test()
        if str(featurizer_name) == "kmer_count":
            return X_train, X_test, y_train, y_test
        return (
            self._one_hot_flatten(X_train["raw_sequence"]),
            self._one_hot_flatten(X_test["raw_sequence"]),
            y_train,
            y_test,
        )

    def load_for_dl(self, *, seed: int) -> tuple[Any, Any, Any, Any]:
        X_train, X_test, y_train, y_test = self._load_train_test()
        return (
            self._one_hot_flatten(X_train["raw_sequence"]),
            self._one_hot_flatten(X_test["raw_sequence"]),
            y_train,
            y_test,
        )
```

### Step 3) Add Hydra dataset config

Create `conf/dataset/my_dataset.yaml`:

```yaml
name: my_dataset
sequence_length: 101
val_size: 0.1
n_nucleotides: 4
data_dir: data/my_dataset/processed
train_filename: train.csv
test_filename: test.csv
sequence_column: raw_sequence
label_column: label
subsample_proportion: ${oc.select:subsample_proportion,1.0}
random_state: ${oc.select:random_state,42}
plugin:
  _target_: src.data.my_dataset_plugin.MyDatasetPlugin
  data_dir: ${..data_dir}
  train_filename: ${..train_filename}
  test_filename: ${..test_filename}
  sequence_column: ${..sequence_column}
  label_column: ${..label_column}
  subsample_proportion: ${..subsample_proportion}
  random_state: ${..random_state}
```

No manual registration is needed when `plugin._target_` is provided.

### Step 4) Add experiment config

Create `conf/config_my_dataset.yaml`:

```yaml
max_depth: 6
min_samples_leaf_fraction: 0.01
random_state: 42
name_experiment: "my_dataset"
subsample_proportion: 1.0

content: "You are an expert in biology and DNA sequences."
params_generation:
  target_name: "my target label"
  dataset_info: "This is my custom DNA classification dataset."

defaults:
  - selector: topk
  - llm_models: generic_user
  - llm: default
  - feature_finder: default
  - dataset: my_dataset
  - splitting_criterion: gini
  - tree: default
  - _self_
```

Then customize prompts and generation params from `conf/config.yaml` as needed.

To use OpenAI instead of the generic/Azure setup:

```yaml
- llm_models: openai_gpt4o_mini
```

### Step 5) Run

```bash
uv run python experiments/deft_tree.py --config-name=config_my_dataset
```

### Step 6) Inspect outputs

- Tree artifact and logs are in `outputs/<date>/<time>/...`

---

## Citation

If you use this repository, please cite the paper:


```bibtex
@misc{huynh2026interpretablednasequenceclassification,
      title={Interpretable DNA Sequence Classification via Dynamic Feature Generation in Decision Trees},
      author={Nicolas Huynh and Krzysztof Kacprzyk and Ryan Sheridan and David Bentley and Mihaela van der Schaar},
      year={2026},
      eprint={2604.12060},
      url={https://arxiv.org/abs/2604.12060},
}
```
