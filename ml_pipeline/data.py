"""Load aligned ESM embeddings and construct sequence-disjoint CV folds."""

import csv
import importlib.util
from pathlib import Path

import numpy as np


def load_extractor(root):
    spec = importlib.util.spec_from_file_location("esm2_extractor", root / "ESM-2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_features(path, input_path=None, extractor=None):
    with np.load(path, allow_pickle=False) as arrays:
        required = {"X", "y", "sequences", "excel_rows"}
        if not required.issubset(arrays.files):
            raise ValueError(f"Feature file requires arrays: {sorted(required)}")
        X, y = arrays["X"], arrays["y"]
        sequences, rows = arrays["sequences"], arrays["excel_rows"]
    if X.ndim != 2 or X.shape[1] != 1280 or not len(X):
        raise ValueError("Expected nonempty ESM-2 features of shape (N, 1280).")
    if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
        raise ValueError("Features must be finite numeric values.")
    n = len(X)
    if any(a.shape != (n,) for a in (y, sequences, rows)):
        raise ValueError("Feature, label, sequence and Excel-row arrays are not aligned.")
    if set(np.unique(y)) != {1, 2, 3}:
        raise ValueError("All three labels (1, 2, 3) must be present.")
    if sequences.dtype.kind != "U" or any(not s for s in sequences):
        raise ValueError("Sequences must be nonempty Unicode strings.")
    if not np.issubdtype(rows.dtype, np.integer) or len(np.unique(rows)) != n:
        raise ValueError("Excel row numbers must be unique integers.")
    # Check every cached row, not just file modification times.
    if input_path is not None and extractor is not None:
        accepted, _ = extractor.read_input(input_path)
        expected = [(int(r), str(s), int(label)) for r, s, label in accepted]
        actual = list(zip(rows.tolist(), sequences.tolist(), y.astype(int).tolist()))
        if actual != expected:
            raise ValueError("Cached features do not match the input XLSX. "
                             "Run main.py --reextract to rebuild them.")
    return X.astype(np.float32), y.astype(np.int64) - 1, sequences, rows


def audit_labels(y, sequences, rows, output, policy):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    labels_by_sequence = {}
    for sequence, label in zip(sequences, y):
        labels_by_sequence.setdefault(str(sequence), set()).add(int(label) + 1)
    conflicts = {s: labels for s, labels in labels_by_sequence.items() if len(labels) > 1}
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "duplicated"
    sheet.append(["sequence", "label", "excel_row", "all_observed_labels"])
    for row, sequence, label in zip(rows, sequences, y):
        if sequence in conflicts:
            sheet.append([str(sequence), int(label) + 1, int(row),
                          ";".join(map(str, sorted(conflicts[sequence])))])
    for cell in sheet[1]:
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="243746")
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.column_dimensions["A"].width = 70
    sheet.column_dimensions["B"].width = 10
    sheet.column_dimensions["C"].width = 14
    sheet.column_dimensions["D"].width = 24
    workbook.save(output / "duplicated.xlsx")
    workbook.close()
    with (output / "label_conflicts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["excel_row", "sequence", "label", "all_observed_labels"])
        for row, sequence, label in zip(rows, sequences, y):
            if sequence in conflicts:
                writer.writerow([row, sequence, label + 1,
                                 ";".join(map(str, sorted(conflicts[sequence])))])
    keep = np.asarray([not (policy == "drop" and s in conflicts) for s in sequences])
    stats = {
        "input_records": len(y), "unique_sequences": len(labels_by_sequence),
        "conflicting_sequences": len(conflicts),
        "conflicting_rows": sum(s in conflicts for s in sequences),
        "conflict_policy": policy, "excluded_conflicting_rows": int((~keep).sum()),
    }
    return keep, stats


def make_splits(y, groups, folds, seed):
    from sklearn.model_selection import StratifiedGroupKFold

    if folds < 2:
        raise ValueError("At least two CV folds are required.")
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Each of the three classes must remain after filtering.")
    for label in range(3):
        if len(np.unique(groups[y == label])) < folds:
            raise ValueError(f"Label {label + 1} has fewer than {folds} unique sequences.")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    seen = np.zeros(len(y), dtype=int)
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Sequence leakage between train and test.")
        if set(y[train]) != {0, 1, 2} or set(y[test]) != {0, 1, 2}:
            raise ValueError("A fold is missing a class; use fewer folds or another seed.")
        seen[test] += 1
    if not np.all(seen == 1):
        raise ValueError("Each row must appear in exactly one held-out fold.")
    return splits


def make_holdout(y, groups, seed):
    """Reserve ~10% by class, keeping every exact sequence in one partition."""
    from sklearn.model_selection import train_test_split

    unique, inverse = np.unique(groups, return_inverse=True)
    labels = [np.unique(y[inverse == i]) for i in range(len(unique))]
    if all(len(values) == 1 for values in labels):
        train_groups, test_groups = train_test_split(
            np.arange(len(unique)), test_size=0.1, random_state=seed,
            stratify=np.asarray([values[0] for values in labels]))
        train = np.flatnonzero(np.isin(inverse, train_groups))
        test = np.flatnonzero(np.isin(inverse, test_groups))
    else:
        # --conflicts keep can contain multiple labels in the same group.
        train, test = make_splits(y, groups, 10, seed)[0]
    if set(y[train]) != {0, 1, 2} or set(y[test]) != {0, 1, 2}:
        raise ValueError("Each holdout partition must contain all three labels.")
    if set(groups[train]) & set(groups[test]):
        raise ValueError("Sequence leakage into independent test.")
    return train, test
