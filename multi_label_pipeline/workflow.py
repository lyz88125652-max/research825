"""Stage-separated research workflow. Holdout evaluation is an explicit final step."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from . import LABEL_NAMES
from .data import frozen_splits, read_input, load_features
from .evaluation import evaluate_cv, save_predictions, write_csv, write_json
from .ensemble import rank_subsets, save_search, candidate_roles, average
from .stacking import fit_stacker
from .nested_evaluation import run_nested
from .metrics import validate_probabilities


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()


def code_hash(root):
    digest = hashlib.sha256()
    for file in sorted([root / "main_multi_label.py", root / "ESM-2.py", *(root / "multi_label_pipeline").glob("*.py")]):
        digest.update(str(file.relative_to(root)).encode()); digest.update(file.read_bytes())
    return digest.hexdigest()


def persist_args(args):
    return {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}


def restore_args(args):
    config = json.loads((args.run_dir / "run_config.json").read_text())
    values = config["arguments"]
    for key in ["input", "features", "output_dir", "split_file", "run_dir", "reuse_features"]:
        if values.get(key) is not None: values[key] = Path(values[key])
    values.update(stage=args.stage, run_dir=args.run_dir, prepare_only=False)
    return argparse.Namespace(**values), config


def cache_trainer(name, trainer):
    """Resume a completed fit only if its data, seed and fit parameters match."""
    def cached(X, y, X_eval, groups, output, args, seed):
        output.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        for values in [X, y, X_eval, groups]:
            array = np.asarray(values)
            digest.update(str(array.shape).encode()); digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        settings = persist_args(args)
        for key in ["stage", "run_dir", "output_dir", "prepare_only"]: settings.pop(key, None)
        digest.update(json.dumps([name, seed, settings], sort_keys=True).encode())
        fingerprint = digest.hexdigest()
        path = output / "prediction_cache.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if saved["fingerprint"].item() != fingerprint: raise ValueError("Training cache mismatch.")
                p, details = saved["p"], json.loads(saved["details"].item())
            return validate_probabilities(np.zeros((len(X_eval),3)), p), details
        try:
            p, details = trainer(X, y, X_eval, groups, output, args, seed)
            p = validate_probabilities(np.zeros((len(X_eval),3)), p)
            temporary = output / "prediction_cache.tmp.npz"
            np.savez_compressed(temporary, p=p, fingerprint=np.asarray(fingerprint), details=np.asarray(json.dumps(details)))
            temporary.replace(path)
            return p, details
        except Exception as exc:
            write_json(output / "failure.json", {"model": name, "seed": seed, "error": str(exc)})
            raise
    return cached


def read_development(output, y, sequences, rows, names):
    path = output / "development_predictions.npz"
    with np.load(path, allow_pickle=False) as saved:
        for key, expected in [("y", y), ("sequences", sequences), ("source_rows", rows), ("model_names", names)]:
            if not np.array_equal(saved[key], expected): raise ValueError("Development probability alignment mismatch.")
        p, folds = saved["p"], saved["folds"]
    from .ensemble import validate_tensor
    return validate_tensor(y, p, names), folds


def run_cv(X, y, sequences, rows, splits, output, args, trainers):
    from .research_reporting import workbook, final_plots, neural_plots
    summaries, predictions = [], []
    for name in args.models:
        summaries.append(evaluate_cv(name, trainers[name], X, y, sequences, rows, splits, output, args))
        with (output / name / "oof_predictions.csv").open(newline="") as handle: records = list(csv.DictReader(handle))
        if [int(record["source_row"]) for record in records] != rows.tolist(): raise ValueError("OOF row mismatch.")
        predictions.append(np.array([[float(record["prob_" + label]) for label in LABEL_NAMES] for record in records]))
    fold_ids = np.zeros(len(y), dtype=int)
    for fold, (_, valid) in enumerate(splits, 1): fold_ids[valid] = fold
    np.savez_compressed(output / "development_predictions.npz", p=np.stack(predictions, axis=1), y=y,
                        sequences=sequences, source_rows=rows, model_names=np.asarray(args.models), folds=fold_ids)
    write_csv(output / "model_comparison_cv.csv", list(summaries[0]), summaries)
    workbook(output / "development_summary.xlsx", {"CV": summaries}, ["Development CV only; holdout not accessed."])
    final_plots(output, {name:(y, p) for name,p in zip(args.models,predictions)}, prefix="oof_")
    neural_plots(output,args.models)


def search(output, y, sequences, rows, args):
    from .research_reporting import plot_search, workbook
    p, _ = read_development(output, y, sequences, rows, args.models)
    ranking = rank_subsets(y, p, args.models, args.threshold)
    by_k = save_search(output / "ensemble", ranking, args.models, args.top_k)
    plot_search(output / "ensemble", ranking, by_k)
    workbook(output / "ensemble/summary.xlsx", {"Ranking": ranking, "By_model_count": by_k},
             ["OOF search ranking is selection-biased. Use nested CV and the locked final test for evaluation."])
    return ranking


def select_recipe(output, y, sequences, rows, args):
    read_development(output, y, sequences, rows, args.models)
    with (output / "ensemble/ranking.csv").open(newline="") as handle:
        ranking = list(csv.DictReader(handle))
    # Ranking order is already fixed by the ensemble stage; do not re-sort strings.
    roles = {f"top_{i:02}": row["candidate"].split("+") for i, row in enumerate(ranking[:args.top_k],1)}
    roles["best_single"] = next(row["candidate"].split("+") for row in ranking if int(row["model_count"]) == 1)
    roles["all_models"] = args.models
    nested = json.loads((output / "nested/summary.json").read_text())
    candidates = [row for row in nested if row["role"] in roles]
    if not candidates: raise ValueError("No nested comparison available.")
    winner = min(candidates, key=lambda row: (-row["f1_macro_mean"],
        -(row["average_precision_macro_mean"] if row["average_precision_macro_mean"] is not None else -1),
        row["mean_model_count"], row["role"], row["method"]))
    recipe = {"schema": 1, "selected": {"role": winner["role"], "method": winner["method"],
                                         "models": roles[winner["role"]]},
              "baselines": {"best_single": roles["best_single"], "all_models": roles["all_models"]},
              "threshold": args.threshold, "meta_c": args.meta_c, "meta_class_weight": args.meta_class_weight,
              "selection_rule": "nested mean Macro-F1, mean Macro AP, mean model count, role and method",
              "test_used": False,
              "artifacts": {name: sha256(output / name) for name in ["run_config.json", "splits.json",
                  "development_predictions.npz", "ensemble/ranking.csv", "nested/summary.json"]}}
    if (output / "final_access.json").exists(): raise ValueError("Cannot reselect after final test access.")
    write_json(output / "final_recipe.json", recipe)
    return recipe


def run_final(X, y, sequences, rows, train, test, output, args, trainers):
    """No selection uses holdout labels. Require a sealed recipe and one-time access marker."""
    from .research_reporting import final_plots, workbook
    recipe = json.loads((output / "final_recipe.json").read_text())
    for path, digest in recipe["artifacts"].items():
        if sha256(output / path) != digest: raise ValueError("Final recipe artifact changed: " + path)
    if recipe["threshold"] != args.threshold or recipe["meta_c"] != args.meta_c or recipe["meta_class_weight"] != args.meta_class_weight:
        raise ValueError("Final recipe settings changed.")
    p, _ = read_development(output, y[train], sequences[train], rows[train], args.models)
    definitions = {"selected": recipe["selected"],
                   **{role: {"method": "voting", "models": subset} for role, subset in recipe["baselines"].items()}}
    # Fit the meta-model before touching holdout features or labels.
    meta, columns = None, None
    if definitions["selected"]["method"] == "stacking":
        meta, columns = fit_stacker(p, y[train], args.models, definitions["selected"]["models"], args, args.seed)
    marker = output / "final_access.json"
    with marker.open("x", encoding="utf-8") as handle:
        json.dump({"status": "started", "recipe_sha256": sha256(output / "final_recipe.json"),
                   "timestamp": datetime.now().astimezone().isoformat()}, handle, indent=2)
    destination = output / "final"; destination.mkdir()
    try:
        base_p = np.empty((len(test), len(args.models), 3))
        for index, name in enumerate(args.models):
            folder = destination / "base_models" / name; folder.mkdir(parents=True)
            values, details = trainers[name](X[train], y[train], X[test], sequences[train], folder, args, args.seed)
            base_p[:, index] = validate_probabilities(y[test], values)
            write_json(folder / "training.json", details)
        results, plotted = [], {}
        for role, definition in definitions.items():
            if role == "selected" and meta is not None:
                values = meta.predict_proba(base_p[:, columns], definition["models"])
            else: values = average(base_p, args.models, definition["models"])
            folder = destination / role; folder.mkdir()
            metrics, _ = save_predictions(folder, "test_", y[test], values, sequences[test], rows[test], args.threshold)
            results.append({"model": role, "method": definition["method"], "candidate": "+".join(definition["models"]), **metrics})
            plotted[role] = y[test], values
        write_csv(destination / "comparison.csv", list(results[0]), results)
        workbook(destination / "summary.xlsx", {"Final_test": results}, ["Frozen selected procedure and prespecified baselines. Do not select again on this table."])
        final_plots(destination, plotted)
        write_json(marker, {"status": "complete", "recipe_sha256": sha256(output / "final_recipe.json"),
                            "test_rows": len(test), "timestamp": datetime.now().astimezone().isoformat()})
    except Exception as exc:
        write_json(marker, {"status": "failed_after_access", "error": str(exc),
                            "interpretation": "Do not automatically rerun or reselect using this holdout."})
        raise


def run(args, root, registry, supplied_trainers=None):
    from .research_reporting import plot_nested, workbook, plot_length
    from .length_baseline import analyze_lengths, error_diversity
    if args.run_dir:
        args, config = restore_args(args)
        output = args.run_dir
        if config["code_sha256"] != code_hash(root): raise ValueError("Code changed since run creation; start a new run.")
        if config["input_sha256"] != sha256(args.input): raise ValueError("Input changed since run creation.")
    else:
        if args.stage not in {"prepare", "cv", "all"}: raise ValueError("This stage requires --run-dir.")
        if args.reextract or args.reuse_features or args.no_feature_reuse:
            raise ValueError("Run ESM-2.py separately to prepare features before starting the experiment.")
        if args.split_file is None:
            args.split_file = root / f"data/multi_label/splits_seed{args.seed}_folds{args.folds}.json"
        output = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True)
        args.run_dir = output
        config = None
    extractor, records, skipped = read_input(args.input, root)
    y = np.asarray([record[2] for record in records], dtype=np.int64)
    sequences = np.asarray([record[1] for record in records]); rows = np.asarray([record[0] for record in records])
    manifest_path = output / "splits.json" if config else args.split_file
    train, test, splits, manifest = frozen_splits(manifest_path, records, args.folds, args.seed, args.test_fraction)
    if config is None:
        write_json(output / "splits.json", manifest)
        versions = {}
        for package in ["numpy", "scikit-learn", "torch", "transformers", "iterative-stratification", "lightgbm", "catboost", "xgboost", "matplotlib", "openpyxl"]:
            try: versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: pass
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
        config = {"schema": 2, "arguments": persist_args(args), "input_sha256": sha256(args.input),
                  "code_sha256": code_hash(root), "split_sha256": sha256(output / "splits.json"),
                  "features_sha256": None, "versions": versions, "commit": commit.stdout.strip() or None,
                  "label_names": list(LABEL_NAMES), "training_rows": len(train), "test_rows": len(test),
                  "label_policy": "recorded positive=1; negative/unrecorded/unknown=0", "threshold": args.threshold}
        config["reference_source_sha256"] = {
            name: sha256(root/"data"/name) for name in ["antioxidant_peptides.csv", "naturalAMPs_APD2024a.csv", "poisonous.csv"]
            if args.input.resolve() == (root/"data/multi-label.csv").resolve() and (root/"data"/name).exists()}
        write_json(output / "run_config.json", config)
        extractor.write_skipped(output / "skipped_sequences.csv", skipped, "multilabel")
        write_json(output / "data_audit.json", {"input_records": len(records)+len(skipped), "accepted_records": len(records),
            "unsupported_records": len(skipped), "retained_multi_activity_records": int((y.sum(1)>1).sum()),
            "label_names": list(LABEL_NAMES), "positive_counts": dict(zip(LABEL_NAMES,y.sum(0).tolist())),
            "combinations": dict(Counter("".join(map(str,target)) for target in y.tolist()))})
        fields = ["source_row", "sequence", *LABEL_NAMES]
        def assignment(index):
            return {"source_row": int(rows[index]), "sequence": str(sequences[index]), **dict(zip(LABEL_NAMES,y[index].tolist()))}
        training_set = set(train)
        write_csv(output / "holdout_assignments.csv", [*fields,"split"],
                  [{**assignment(i),"split":"training" if i in training_set else "test"} for i in range(len(y))])
        write_csv(output / "fold_assignments.csv", [*fields,"fold"],
                  [{**assignment(train[i]),"fold":fold} for fold,(_,valid) in enumerate(splits,1) for i in valid])
    elif sha256(output / "splits.json") != config["split_sha256"]:
        raise ValueError("Run split manifest was changed.")
    print(f"Run: {output}\nAccepted={len(y)} training={len(train)} locked_test={len(test)}", flush=True)
    if args.stage == "prepare": return output
    if not args.features.exists(): raise ValueError("Features missing. Run ESM-2.py first; no automatic extraction.")
    metadata_path = args.features.with_suffix(".metadata.json")
    if not metadata_path.exists(): raise ValueError("Feature metadata missing; extract with ESM-2.py first.")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("model") != "facebook/esm2_t33_650M_UR50D" or metadata.get("layer") != 33 or metadata.get("features") != 1280:
        raise ValueError("Feature metadata does not describe the planned ESM-2 650M layer 33 embeddings.")
    if metadata.get("pooling") != "mean over residue tokens; exclude BOS, EOS and padding":
        raise ValueError("Feature pooling metadata mismatch.")
    if metadata.get("reused_records", 0) and not args.allow_reused_features:
        raise ValueError("Plan requires fresh ESM-2 extraction; this cache reports reused records. Use ESM-2.py or explicitly --allow-reused-features for a separate exploratory run.")
    if metadata.get("input_sha256") and metadata["input_sha256"] != sha256(args.input):
        raise ValueError("Feature metadata input hash mismatch.")
    feature_hash = sha256(args.features)
    if config["features_sha256"] is not None and feature_hash != config["features_sha256"]:
        raise ValueError("Features changed since run creation.")
    metadata_hash = sha256(metadata_path)
    if config.get("feature_metadata_sha256") and metadata_hash != config["feature_metadata_sha256"]:
        raise ValueError("Feature metadata changed since run creation.")
    if config["features_sha256"] is None:
        config.update(features_sha256=feature_hash, feature_metadata_sha256=metadata_hash, feature_metadata=metadata)
        write_json(output / "run_config.json", config)
    X, _, _, _ = load_features(args.features, records)
    if supplied_trainers is None:
        supplied_trainers = {}
        if args.stage in {"cv","nested","all","final"}:
            supplied_trainers = {name: importlib.import_module("multi_label_pipeline."+registry[name]).train_predict for name in args.models}
        if args.stage in {"bias", "all"}:
            supplied_trainers["length_lr"] = importlib.import_module("multi_label_pipeline.logistic_regression").train_predict
    trainers = {name: cache_trainer(name, trainer) for name,trainer in supplied_trainers.items()}
    stages = ["cv", "ensemble", "nested", "bias", "select"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if (output / "final_access.json").exists() and stage != "final": raise ValueError("Development is sealed after final access.")
        print("Stage: "+stage, flush=True)
        try:
            if stage == "cv": run_cv(X[train],y[train],sequences[train],rows[train],splits,output,args,trainers)
            elif stage == "ensemble": search(output,y[train],sequences[train],rows[train],args)
            elif stage == "nested":
                summaries = run_nested(X[train],y[train],sequences[train],rows[train],args.models,splits,output/"nested",args,trainers)
                plot_nested(output/"nested")
                workbook(output/"nested/summary.xlsx",{"Outer_CV":summaries},["Ranks refer to selection procedures; subsets may differ by outer fold."])
            elif stage == "select": select_recipe(output,y[train],sequences[train],rows[train],args)
            elif stage == "final": run_final(X,y,sequences,rows,train,test,output,args,trainers)
            elif stage == "bias":
                p,_ = read_development(output,y[train],sequences[train],rows[train],args.models)
                lengths = analyze_lengths(y[train],sequences[train],rows[train],p,args.models,splits,output/"bias",args,trainers["length_lr"])
                error_diversity(y[train],p,args.models,args.threshold,output/"bias")
                plot_length(output/"bias",lengths,y[train])
            write_json(output/f"stage_{stage}.json",{"status":"complete","holdout_used":stage=="final"})
        except Exception as exc:
            write_json(output/f"stage_{stage}.json",{"status":"failed","error":str(exc)})
            raise
    return output
