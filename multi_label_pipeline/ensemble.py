"""Enumerate all nonempty subsets using development OOF probabilities only."""

from itertools import combinations
import numpy as np
from .metrics import score, validate_probabilities
from .evaluation import write_csv, write_json


def validate_tensor(y, probabilities, names):
    p = np.asarray(probabilities, dtype=float)
    if len(set(names)) != len(names) or not names:
        raise ValueError("Expected distinct model names.")
    if p.shape != (len(y), len(names), 3): raise ValueError("Expected (samples, models, 3) probabilities.")
    for index in range(len(names)): validate_probabilities(y, p[:, index])
    return p


def average(probabilities, names, subset):
    if not subset or len(set(subset)) != len(subset) or not set(subset).issubset(names):
        raise ValueError("Invalid ensemble subset.")
    return np.mean(probabilities[:, [names.index(name) for name in subset], :], axis=1)


def ranking_key(row):
    return (-row["f1_macro"], -(row["average_precision_macro"] if row["average_precision_macro"] is not None else -1),
            row["model_count"], row["candidate"])


def rank_subsets(y, probabilities, names, threshold=.5):
    p = validate_tensor(y, probabilities, names)
    rows = []
    for count in range(1, len(names) + 1):
        for subset in combinations(names, count):
            metrics, _ = score(y, average(p, names, subset), threshold)
            rows.append({"candidate": "+".join(subset), "model_count": count, **metrics})
    rows.sort(key=ranking_key)
    return [{"rank": rank, **row} for rank, row in enumerate(rows, 1)]


def candidate_roles(ranking, names, top_k):
    roles = {f"top_{rank:02}": row["candidate"].split("+")
             for rank, row in enumerate(ranking[:top_k], 1)}
    roles["best_single"] = next(row["candidate"].split("+") for row in ranking if row["model_count"] == 1)
    roles["all_models"] = list(names)
    return roles


def save_search(output, ranking, names, top_k):
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "ranking.csv", list(ranking[0]), ranking)
    by_k = []
    for count in range(1, len(names) + 1):
        values = np.array([row["f1_macro"] for row in ranking if row["model_count"] == count])
        by_k.append({"model_count": count, "combinations": len(values), "best_macro_f1": float(values.max()),
                     "median_macro_f1": float(np.median(values)), "mean_macro_f1": float(values.mean())})
    write_csv(output / "by_model_count.csv", list(by_k[0]), by_k)
    write_json(output / "selection.json", {
        "candidate_count": len(ranking), "roles": candidate_roles(ranking, names, top_k),
        "selection_rule": "OOF Macro-F1 desc, Macro AP desc, fewer models, candidate name",
        "interpretation": "development search scores; not an unbiased performance estimate",
    })
    return by_k
