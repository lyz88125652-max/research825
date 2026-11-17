"""Multi-label input/cache validation and iterative stratification."""

import importlib.util
from pathlib import Path

import numpy as np

from . import LABEL_NAMES


def load_extractor(root):
    spec = importlib.util.spec_from_file_location("esm2_shared_extractor", root / "ESM-2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_input(path, root):
    extractor = load_extractor(root)
    task, records, skipped = extractor.read_dataset(path, "multilabel")
    if not records:
        raise ValueError("No supported sequences remain.")
    return extractor, records, skipped


def load_features(path, records):
    with np.load(path, allow_pickle=False) as cache:
        required = {"X", "y", "sequences", "source_rows", "label_names", "task"}
        if not required.issubset(cache.files):
            raise ValueError("Multi-label NPZ requires X, y, sequences, source_rows, label_names, task.")
        if cache["task"].item() != "multilabel" or tuple(cache["label_names"].tolist()) != LABEL_NAMES:
            raise ValueError("Feature task or label order is incompatible.")
        X, y, sequences, rows = (cache[k] for k in ["X", "y", "sequences", "source_rows"])
    n = len(records)
    if X.shape != (n, 1280) or not np.issubdtype(X.dtype, np.floating) or not np.isfinite(X).all():
        raise ValueError("Expected finite floating-point features of shape (N, 1280).")
    if y.shape != (n, 3) or not np.isin(y, [0, 1]).all():
        raise ValueError("Targets must be an (N, 3) binary matrix.")
    if sequences.shape != (n,) or rows.shape != (n,) or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("Sequence and source-row arrays must align with features.")
    expected = [(r[0], r[1], tuple(r[2])) for r in records]
    actual = [(int(row), str(seq), tuple(int(v) for v in label))
              for row, seq, label in zip(rows, sequences, y)]
    if actual != expected:
        raise ValueError("Feature cache does not match CSV; use --reextract.")
    if len(set(sequences)) != n:
        raise ValueError("Duplicate sequences must be merged before splitting.")
    return X.astype(np.float32), y.astype(np.int64), sequences, rows


def validate_targets(y, minimum=1):
    if y.ndim != 2 or y.shape[1] != 3 or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected a three-column binary target matrix.")
    for i, label in enumerate(LABEL_NAMES):
        positives = int(y[:, i].sum())
        if min(positives, len(y) - positives) < minimum:
            raise ValueError(f"{label} needs at least {minimum} positive and negative records.")


def make_holdout(y, sequences, seed, fraction=0.1):
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    if len(set(sequences)) != len(y):
        raise ValueError("Expected one row per unique peptide sequence.")
    validate_targets(y, 2)
    split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    train, test = next(split.split(np.zeros(len(y)), y))
    for indices in [train, test]:
        validate_targets(y[indices])
    if set(sequences[train]) & set(sequences[test]):
        raise ValueError("Sequence leakage into holdout.")
    return train, test


def make_splits(y, sequences, folds, seed):
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    if folds < 2 or len(set(sequences)) != len(y):
        raise ValueError("Need >=2 folds and unique input sequences.")
    validate_targets(y, folds)
    split = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = list(split.split(np.zeros(len(y)), y))
    seen = np.zeros(len(y), dtype=int)
    for train, test in result:
        validate_targets(y[train])
        validate_targets(y[test])
        if set(sequences[train]) & set(sequences[test]):
            raise ValueError("Sequence leakage between CV partitions.")
        seen[test] += 1
    if not np.all(seen == 1):
        raise ValueError("Every row must be evaluated once in CV.")
    return result
