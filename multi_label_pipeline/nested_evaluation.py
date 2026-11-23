"""Select candidates and fit meta-models inside each outer training partition."""

import json
import numpy as np
from .data import make_splits
from .ensemble import rank_subsets, candidate_roles, average, save_search
from .stacking import fit_stacker
from .metrics import score, validate_probabilities
from .evaluation import save_predictions, write_csv, write_json


def collect_oof(X, y, sequences, rows, names, splits, output, args, trainers, seed):
    """Return aligned OOF tensor; record the exact training/validation provenance."""
    output.mkdir(parents=True, exist_ok=True)
    probabilities = np.full((len(y), len(names), 3), np.nan)
    fold_ids, seen = np.zeros(len(y), dtype=int), np.zeros(len(y), dtype=int)
    assignments = []
    for fold, (fit, valid) in enumerate(splits, 1):
        if set(sequences[fit]) & set(sequences[valid]): raise ValueError("OOF sequence leakage.")
        seen[valid] += 1; fold_ids[valid] = fold
        assignments.append({"fold": fold, "fit_source_rows": rows[fit].tolist(),
                            "validation_source_rows": rows[valid].tolist()})
        for index, name in enumerate(names):
            folder = output / name / f"fold_{fold}"; folder.mkdir(parents=True, exist_ok=True)
            p, details = trainers[name](X[fit], y[fit], X[valid], sequences[fit], folder, args, seed + fold)
            probabilities[valid, index] = validate_probabilities(y[valid], p)
            metrics, _ = score(y[valid], p, args.threshold)
            write_json(folder / "metrics.json", {"fold": fold, **metrics, "training": details})
    if not np.all(seen == 1): raise ValueError("Incomplete or overlapping OOF coverage.")
    write_json(output / "fold_provenance.json", assignments)
    for index, name in enumerate(names):
        save_predictions(output / name, "oof_", y, probabilities[:, index], sequences, rows, args.threshold, fold_ids)
    return probabilities, fold_ids


def run_nested(X, y, sequences, rows, names, outer_splits, output, args, trainers):
    output.mkdir(parents=True, exist_ok=True)
    comparisons, pools, pair_rows = [], {}, []
    outer_fold_ids = np.zeros(len(y), dtype=int)
    for fold, (fit, valid) in enumerate(outer_splits, 1):
        print(f"Nested outer fold {fold}/{len(outer_splits)}", flush=True)
        folder = output / f"outer_{fold}"; folder.mkdir(exist_ok=True)
        inner = make_splits(y[fit], sequences[fit], args.inner_folds, args.seed + 1000 * fold)
        inner_p, _ = collect_oof(X[fit], y[fit], sequences[fit], rows[fit], names, inner,
                                 folder / "inner", args, trainers, args.seed + 1000 * fold)
        ranking = rank_subsets(y[fit], inner_p, names, args.threshold)
        save_search(folder / "search", ranking, names, args.top_k)
        roles = candidate_roles(ranking, names, args.top_k)
        outer_p = np.empty((len(valid), len(names), 3))
        for index, name in enumerate(names):
            destination = folder / name / "refit"; destination.mkdir(parents=True, exist_ok=True)
            p, details = trainers[name](X[fit], y[fit], X[valid], sequences[fit], destination, args, args.seed + fold)
            outer_p[:, index] = validate_probabilities(y[valid], p)
            write_json(destination / "training.json", details)
        write_json(folder / "partition.json", {"outer_fit_rows": rows[fit].tolist(), "outer_validation_rows": rows[valid].tolist()})
        outer_fold_ids[valid] = fold
        for role, subset in roles.items():
            voting = average(outer_p, names, subset)
            methods = {"voting": voting}
            if role.startswith("top_"):
                stacker, columns = fit_stacker(inner_p, y[fit], names, subset, args, args.seed + fold)
                methods["stacking"] = stacker.predict_proba(outer_p[:, columns], subset)
            for method, p in methods.items():
                key = role + "__" + method
                pools.setdefault(key, np.full(y.shape, np.nan))[valid] = p
                metrics, _ = score(y[valid], p, args.threshold)
                comparisons.append({"fold": fold, "role": role, "method": method,
                                    "candidate": "+".join(subset), "model_count": len(subset), **metrics})
            if "stacking" in methods:
                vote_metrics, _ = score(y[valid], voting, args.threshold)
                stack_metrics, _ = score(y[valid], methods["stacking"], args.threshold)
                pair_rows.append({"fold": fold, "role": role, "candidate": "+".join(subset),
                                  "voting_macro_f1": vote_metrics["f1_macro"],
                                  "stacking_macro_f1": stack_metrics["f1_macro"],
                                  "delta_macro_f1": stack_metrics["f1_macro"] - vote_metrics["f1_macro"]})
    write_csv(output / "fold_comparisons.csv", list(comparisons[0]), comparisons)
    write_csv(output / "paired_differences.csv", list(pair_rows[0]), pair_rows)
    summaries = []
    for key, p in pools.items():
        role, method = key.split("__")
        destination = output / key; destination.mkdir(exist_ok=True)
        pooled, _ = save_predictions(destination, "outer_oof_", y, p, sequences, rows, args.threshold, outer_fold_ids)
        fold_rows = [row for row in comparisons if row["role"] == role and row["method"] == method]
        summary = {"role": role, "method": method,
                   "mean_model_count": float(np.mean([row["model_count"] for row in fold_rows]))}
        for metric in pooled:
            values = [row[metric] for row in fold_rows if row[metric] is not None]
            summary[metric + "_mean"] = float(np.mean(values)) if values else None
            summary[metric + "_std"] = float(np.std(values, ddof=1)) if len(values)>1 else None
            summary[metric + "_valid_folds"] = len(values)
        summaries.append(summary)
    write_csv(output / "summary.csv", list(summaries[0]), summaries)
    write_json(output / "summary.json", summaries)
    write_json(output / "interpretation.json", {
        "unit": "candidate-selection procedure by rank; subsets may differ between outer folds",
        "holdout_used": False, "paired_comparison": "same subset and outer fold for voting and stacking",
        "selection_bias": "Choosing the final procedure on this summary still requires the untouched final test."})
    return summaries
