"""Shared CV, independent-test metrics and predictions for multi-label models."""

import csv
import json
import numpy as np

from . import LABEL_NAMES
from .metrics import score, validate_probabilities


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, fields, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)


def save_predictions(folder, prefix, y, probabilities, sequences, rows, threshold, folds=None):
    p = validate_probabilities(y, probabilities)
    predictions = (p >= threshold).astype(int)
    overall, classes = score(y, p, threshold)
    fields = ["source_row", "sequence"] + (["fold"] if folds is not None else [])
    fields += [f"{kind}_{name}" for kind in ["true", "pred", "prob"] for name in LABEL_NAMES]
    def records():
        for i in range(len(y)):
            record = {"source_row": int(rows[i]), "sequence": str(sequences[i])}
            if folds is not None: record["fold"] = int(folds[i])
            for j, name in enumerate(LABEL_NAMES):
                record[f"true_{name}"] = int(y[i, j])
                record[f"pred_{name}"] = int(predictions[i, j])
                record[f"prob_{name}"] = float(p[i, j])
            yield record
    write_csv(folder / f"{prefix}predictions.csv", fields, records())
    write_csv(folder / f"{prefix}label_metrics.csv", list(classes[0]), classes)
    write_json(folder / f"{prefix}metrics.json", {"n": len(y), "threshold": threshold, **overall})
    return overall, classes


def evaluate_cv(name, trainer, X, y, sequences, rows, splits, output, args):
    folder = output / name; folder.mkdir()
    p = np.full(y.shape, np.nan); folds = np.zeros(len(y), dtype=int)
    results = []
    for fold, (train, test) in enumerate(splits, 1):
        destination = folder / f"fold_{fold}"; destination.mkdir()
        print(f"[{name}] CV fold {fold}/{len(splits)} train={len(train)} validation={len(test)}", flush=True)
        probs, details = trainer(X[train], y[train], X[test], sequences[train],
                                  destination, args, args.seed + fold)
        probs = validate_probabilities(y[test], probs)
        p[test] = probs; folds[test] = fold
        metrics, _ = score(y[test], probs, args.threshold)
        results.append({"fold": fold, "train_size": len(train), "validation_size": len(test), **metrics})
        write_json(destination / "metrics.json", {**results[-1], "training": details})
        print(f"    exact-match={metrics['subset_accuracy']:.4f} macro-F1={metrics['f1_macro']:.4f}", flush=True)
    if np.any(folds == 0): raise ValueError("Incomplete out-of-fold predictions.")
    pooled, _ = save_predictions(folder, "oof_", y, p, sequences, rows, args.threshold, folds)
    write_csv(folder / "fold_metrics.csv", list(results[0]), results)
    summary = {"model": name}
    for key in pooled:
        values = [r[key] for r in results if r[key] is not None]
        summary[key + "_mean"] = float(np.mean(values)) if values else None
        summary[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary[key + "_valid_folds"] = len(values)
    write_json(folder / "summary.json", {"cv_fold_mean_std": summary, "pooled_out_of_fold": pooled})
    return summary


def evaluate_holdout(name, trainer, X, y, sequences, X_test, y_test, test_sequences, test_rows, output, args):
    folder = output / name
    final = folder / "final_model"; final.mkdir()
    print(f"[{name}] final train={len(y)} independent test={len(y_test)}", flush=True)
    p, details = trainer(X, y, X_test, sequences, final, args, args.seed)
    metrics, _ = save_predictions(folder, "test_", y_test, p, test_sequences, test_rows, args.threshold)
    write_json(final / "metrics.json", {"train_size": len(y), "test_size": len(y_test),
                                        **metrics, "training": details})
    return {"model": name, **metrics}
