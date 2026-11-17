"""Multi-label metrics; zeros are the supplied negative/unknown annotations."""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, hamming_loss, jaccard_score,
                             matthews_corrcoef, roc_auc_score)

from . import LABEL_NAMES


def validate_probabilities(y, probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Expected finite independent probabilities of shape (N, 3) in [0, 1].")
    # Deliberately do not normalize the row sums.
    return p


def binary_log_loss(y, probabilities):
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-15, 1 - 1e-15)
    return float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())


def score(y, probabilities, threshold=0.5):
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != 3 or not np.isin(y, [0, 1]).all() or not len(y):
        raise ValueError("Expected nonempty binary targets of shape (N, 3).")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between zero and one.")
    p = validate_probabilities(y, probabilities)
    predicted = (p >= threshold).astype(int)
    classes = []
    for i, label in enumerate(LABEL_NAMES):
        tn, fp, fn, tp = map(int, confusion_matrix(y[:, i], predicted[:, i], labels=[0, 1]).ravel())
        ratio = lambda a, b: float(a / b) if b else None
        both = len(np.unique(y[:, i])) == 2
        sensitivity, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
        classes.append({
            "label": label, "n": len(y), "positives": tp + fn, "negatives": tn + fp,
            "accuracy": float((tp + tn) / len(y)), "sensitivity": sensitivity,
            "specificity": specificity, "precision": ratio(tp, tp + fp) if tp + fp else 0.0,
            "f1": float(f1_score(y[:, i], predicted[:, i], zero_division=0)),
            "mcc": float(matthews_corrcoef(y[:, i], predicted[:, i])),
            "roc_auc": float(roc_auc_score(y[:, i], p[:, i])) if both else None,
            "average_precision": float(average_precision_score(y[:, i], p[:, i])) if both else None,
            "balanced_accuracy": (sensitivity + specificity) / 2 if both else None,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        })
    def mean(key):
        values = [r[key] for r in classes]
        return float(np.mean(values)) if all(v is not None for v in values) else None
    overall = {
        "subset_accuracy": float(accuracy_score(y, predicted)),
        "label_accuracy_mean": float((y == predicted).mean()),
        "hamming_loss": float(hamming_loss(y, predicted)),
        "sensitivity_macro": mean("sensitivity"), "specificity_macro": mean("specificity"),
        "precision_macro": mean("precision"), "f1_macro": mean("f1"),
        "f1_micro": float(f1_score(y, predicted, average="micro", zero_division=0)),
        "f1_samples": float(f1_score(y, predicted, average="samples", zero_division=0)),
        "jaccard_samples": float(jaccard_score(y, predicted, average="samples", zero_division=0)),
        "roc_auc_macro": mean("roc_auc"), "average_precision_macro": mean("average_precision"),
        "balanced_accuracy_macro": mean("balanced_accuracy"), "mcc_macro": mean("mcc"),
        "mcc_micro": float(matthews_corrcoef(y.ravel(), predicted.ravel())),
        "binary_log_loss": binary_log_loss(y, p),
    }
    return overall, classes
