"""Shared metrics and held-out predictions for all classifiers."""

import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path, fields, records):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def metrics(y, probabilities):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, log_loss, matthews_corrcoef,
                                 precision_score, recall_score, roc_auc_score)

    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision_macro": float(precision_score(y, predictions, labels=[0, 1, 2],
                                                  average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, predictions, labels=[0, 1, 2],
                                            average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, predictions, labels=[0, 1, 2],
                                    average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, predictions, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, predictions)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "roc_auc_ovr_macro": float(roc_auc_score(y, probabilities, labels=[0, 1, 2],
                                                  multi_class="ovr", average="macro")),
    }


def evaluate_model(name, train_predict, X, y, sequences, rows, splits, output, args):
    from sklearn.metrics import classification_report, confusion_matrix

    model_output = output / name
    model_output.mkdir()
    probabilities = np.full((len(y), 3), np.nan, dtype=np.float64)
    fold_ids = np.zeros(len(y), dtype=int)
    fold_results = []
    for fold, (train, test) in enumerate(splits, start=1):
        folder = model_output / f"fold_{fold}"
        folder.mkdir()
        print(f"[{name}] fold {fold}/{len(splits)}: train={len(train)} test={len(test)}",
              flush=True)
        probs, details = train_predict(X[train], y[train], X[test], sequences[train],
                                       folder, args, args.seed + fold)
        probs = np.asarray(probs, dtype=np.float64)
        if (probs.shape != (len(test), 3) or not np.isfinite(probs).all()
                or np.any(probs < 0) or np.any(probs > 1)
                or not np.allclose(probs.sum(axis=1), 1, atol=1e-5)):
            raise ValueError(f"{name}: invalid prediction probabilities.")
        # Float32 softmax can sum slightly away from one; normalize for sklearn.
        probs /= probs.sum(axis=1, keepdims=True)
        probabilities[test] = probs
        fold_ids[test] = fold
        result = {"fold": fold, "train_size": len(train), "test_size": len(test),
                  **metrics(y[test], probs)}
        fold_results.append(result)
        write_json(folder / "metrics.json", {**result, "training": details})
        print(f"    accuracy={result['accuracy']:.4f} macro-F1={result['f1_macro']:.4f}",
              flush=True)
    if not np.isfinite(probabilities).all() or np.any(fold_ids == 0):
        raise ValueError("Missing out-of-fold predictions.")
    predictions = probabilities.argmax(axis=1)
    write_csv(model_output / "fold_metrics.csv", list(fold_results[0]), fold_results)
    write_csv(model_output / "predictions.csv",
              ["excel_row", "sequence", "fold", "true_label", "predicted_label",
               "prob_label_1", "prob_label_2", "prob_label_3"],
              ({"excel_row": int(rows[i]), "sequence": str(sequences[i]),
                "fold": int(fold_ids[i]), "true_label": int(y[i]) + 1,
                "predicted_label": int(predictions[i]) + 1,
                **{f"prob_label_{c + 1}": float(probabilities[i, c]) for c in range(3)}}
               for i in range(len(y))))
    matrix = confusion_matrix(y, predictions, labels=[0, 1, 2])
    write_csv(model_output / "confusion_matrix.csv", ["true_label", "pred_1", "pred_2", "pred_3"],
              ({"true_label": c + 1, **{f"pred_{j + 1}": int(matrix[c, j]) for j in range(3)}}
               for c in range(3)))
    report = classification_report(y + 1, predictions + 1, labels=[1, 2, 3],
                                   output_dict=True, zero_division=0)
    write_json(model_output / "classification_report.json", report)
    pooled = metrics(y, probabilities)
    summary = {"model": name}
    for key in pooled:
        values = [result[key] for result in fold_results]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values, ddof=1))
    write_json(model_output / "summary.json", {
        "fold_mean_std": summary, "pooled_out_of_fold": pooled,
    })
    return summary
