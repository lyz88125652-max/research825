"""Training-only length baseline and sensitivity analyses of development OOF."""

from itertools import combinations
import numpy as np
from .ensemble import validate_tensor
from .evaluation import evaluate_cv, write_csv, write_json
from .metrics import score


def analyze_lengths(y, sequences, rows, probabilities, names, splits, output, args, trainer):
    output.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(sequence) for sequence in sequences], dtype=float)
    evaluate_cv("length_lr", trainer, lengths[:, None], y, sequences, rows, splits, output, args)
    if args.length_range:
        lower, upper = args.length_range
    else:
        lower = int(max(lengths[y[:, c] == 1].min() for c in range(3)))
        upper = int(min(lengths[y[:, c] == 1].max() for c in range(3)))
    mask = (lengths >= lower) & (lengths <= upper)
    common = []
    if mask.any():
        for index, name in enumerate(names):
            metrics, _ = score(y[mask], probabilities[mask, index], args.threshold)
            common.append({"model": name, "n": int(mask.sum()), "min_length": lower,
                           "max_length": upper, **metrics})
        write_csv(output / "common_length_oof_metrics.csv", list(common[0]), common)
    distributions = []
    for column, label in enumerate(["antioxidant", "antimicrobial", "toxic"]):
        values = lengths[y[:, column] == 1]
        distributions.append({"label": label, "n": len(values), "minimum": int(values.min()),
                              "median": float(np.median(values)), "mean": float(values.mean()),
                              "maximum": int(values.max())})
    write_csv(output / "length_distribution.csv", list(distributions[0]), distributions)
    write_json(output / "interpretation.json", {
        "holdout_used": False, "common_length_range": [lower, upper], "common_length_n": int(mask.sum()),
        "range_origin": "explicit CLI" if args.length_range else "intersection of positive-label length ranges in training only",
        "common_range_method": "restrict existing development OOF evaluation; base models not refit on range-only data",
        "zero_policy": "unrecorded/unknown treated as zero; not confirmed biological negatives",
        "source_classifier": "Not run: source memberships define the activity labels and are confounded with them.",
    })
    return lengths


def error_diversity(y, probabilities, names, threshold, output):
    p = validate_tensor(y, probabilities, names)
    errors = (p >= threshold) != y[:, None, :]
    comparisons = []
    for first, second in combinations(range(len(names)), 2):
        a, b = errors[:, first].ravel(), errors[:, second].ravel()
        correlation = float(np.corrcoef(a.astype(float), b.astype(float))[0, 1]) if a.std() and b.std() else None
        comparisons.append({"model_a": names[first], "model_b": names[second],
                            "binary_error_agreement": float(np.mean(a == b)),
                            "binary_prediction_disagreement": float(np.mean(a != b)),
                            "error_correlation": correlation})
    if comparisons: write_csv(output / "model_error_diversity.csv", list(comparisons[0]), comparisons)
