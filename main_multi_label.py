#!/usr/bin/env python3
"""Run shared ESM-2 encoding, multi-label CV and independent-test evaluation.

python main_multi_label.py
python main_multi_label.py --prepare-only
python main_multi_label.py --models rf lightgbm
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODEL_REGISTRY = {
    "cnn": "cnn", "rf": "random_forest", "lightgbm": "lightgbm_model",
    "lr": "logistic_regression", "mlp": "mlp", "chain": "classifier_chain",
    "catboost": "catboost_model", "xgboost": "xgboost_model", "svm": "svm",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=ROOT / "data/multi-label.csv")
    p.add_argument("--features", type=Path, default=ROOT / "data/multi_label/multi_label_esm2_650M.npz")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results_multi_label")
    p.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY), default=list(MODEL_REGISTRY))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--esm-batch-size", type=int, default=8)
    p.add_argument("--reextract", action="store_true")
    p.add_argument("--reuse-features", type=Path, help="Optional compatible sequence-embedding cache.")
    p.add_argument("--no-feature-reuse", action="store_true", help="Disable automatic old-cache reuse.")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--cnn-epochs", type=int, default=30)
    p.add_argument("--cnn-batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--mlp-epochs", type=int, default=30)
    p.add_argument("--mlp-batch-size", type=int, default=64)
    p.add_argument("--linear-c", type=float, default=1.0, help="Inverse L2 strength for LR and chain.")
    p.add_argument("--svm-c", type=float, default=1.0)
    p.add_argument("--chain-folds", type=int, default=3)
    p.add_argument("--chain-order", nargs=3, type=int, default=[0, 1, 2],
                   help="Permutation of 0=antioxidant, 1=antimicrobial, 2=toxic.")
    args = p.parse_args(argv)
    for option in ["folds", "esm_batch_size", "trees", "cnn_epochs", "cnn_batch_size", "patience",
                   "mlp_epochs", "mlp_batch_size", "chain_folds"]:
        if getattr(args, option) < (2 if option in {"folds", "chain_folds"} else 1): p.error(f"Invalid {option}.")
    if not 0 < args.test_fraction < 0.5 or not 0 < args.threshold < 1:
        p.error("Use 0 < test-fraction < 0.5 and 0 < threshold < 1.")
    if args.jobs == 0 or args.learning_rate <= 0 or args.seed < 0:
        p.error("jobs must be nonzero; learning-rate positive; seed nonnegative.")
    if args.features.suffix.lower() != ".npz": p.error("Features must be an .npz file.")
    if args.no_feature_reuse and args.reuse_features: p.error("Conflicting feature-reuse options.")
    import math
    if any(not math.isfinite(value) or value <= 0 for value in [args.linear_c, args.svm_c]):
        p.error("linear-c and svm-c must be finite and positive.")
    if sorted(args.chain_order) != [0, 1, 2]: p.error("chain-order must be a permutation of 0 1 2.")
    args.models = list(dict.fromkeys(args.models))
    return args


def dependency_check(args):
    modules = ["numpy", "sklearn", "iterstrat"]
    if not args.prepare_only:
        modules += ["openpyxl", "joblib", "matplotlib"]
        if {"cnn", "mlp"} & set(args.models): modules.append("torch")
        if "lightgbm" in args.models: modules.append("lightgbm")
        for module in ["catboost", "xgboost"]:
            if module in args.models: modules.append(module)
    failures = []
    for module in modules:
        try: importlib.import_module(module)
        except (ImportError, OSError) as exc: failures.append(f"{module}: {exc}")
    if failures:
        raise RuntimeError("Dependencies unavailable:\n" + "\n".join(failures)
                           + f"\nInstall: {sys.executable} -m pip install -r {ROOT / 'requirements_multi_label.txt'}"
                           + "\nFor a macOS LightGBM libomp error: brew install libomp")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    try:
        dependency_check(args)
        import numpy as np
        from multi_label_pipeline import LABEL_NAMES
        from multi_label_pipeline.data import read_input, load_features, make_holdout, make_splits
        from multi_label_pipeline.evaluation import write_json, write_csv, evaluate_cv, evaluate_holdout

        extractor, records, skipped = read_input(args.input, ROOT)
        y = np.asarray([r[2] for r in records], dtype=np.int64)
        sequences = np.asarray([r[1] for r in records])
        rows = np.asarray([r[0] for r in records])
        train, test = make_holdout(y, sequences, args.seed, args.test_fraction)
        splits = make_splits(y[train], sequences[train], args.folds, args.seed)
        output = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True)
        extractor.write_skipped(output / "skipped_sequences.csv", skipped, "multilabel")
        audit = {"input_records": len(records) + len(skipped), "accepted_records": len(records),
                 "unsupported_records": len(skipped), "retained_multi_activity_records": int((y.sum(1) > 1).sum()),
                 "label_names": list(LABEL_NAMES), "positive_counts": dict(zip(LABEL_NAMES, y.sum(0).tolist())),
                 "combinations": dict(Counter(",".join(map(str, label)) for label in y.tolist())),
                 "zero_meaning": "negative or unknown / not recorded; treated as 0 in this baseline"}
        write_json(output / "data_audit.json", audit)
        flags = ["source_row", "sequence", *LABEL_NAMES]
        def assignments(indices, field, values):
            for i, value in zip(indices, values):
                yield {"source_row": int(rows[i]), "sequence": str(sequences[i]),
                       **dict(zip(LABEL_NAMES, y[i].tolist())), field: value}
        parts = np.full(len(y), "training", dtype=object); parts[test] = "test"
        write_csv(output / "holdout_assignments.csv", [*flags, "split"],
                  assignments(np.arange(len(y)), "split", parts))
        fold_ids = np.zeros(len(train), dtype=int)
        for fold, (_, validation) in enumerate(splits, 1): fold_ids[validation] = fold
        write_csv(output / "fold_assignments.csv", [*flags, "fold"], assignments(train, "fold", fold_ids.tolist()))
        print(f"Accepted={len(y)} skipped={len(skipped)} training={len(train)} test={len(test)}", flush=True)
        if args.prepare_only:
            print(f"Prepared: {output}"); return 0

        if args.reextract or not args.features.exists():
            command = [sys.executable, str(ROOT / "ESM-2.py"), "--task", "multilabel", "--input", str(args.input),
                       "--output", str(args.features), "--batch-size", str(args.esm_batch_size), "--device", args.device]
            reuse = args.reuse_features
            old_cache = ROOT / "data/multi_class/training_cv_esm2_650M.npz"
            if reuse is None and not args.no_feature_reuse and old_cache.exists(): reuse = old_cache
            if reuse is not None: command += ["--reuse-features", str(reuse)]
            subprocess.run(command, check=True)
        X, loaded_y, loaded_sequences, loaded_rows = load_features(args.features, records)
        assert np.array_equal(y, loaded_y) and np.array_equal(sequences, loaded_sequences) and np.array_equal(rows, loaded_rows)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        versions = {}
        for package in ["numpy", "scikit-learn", "iterative-stratification", "torch", "lightgbm", "catboost", "xgboost", "transformers", "openpyxl", "matplotlib"]:
            try: versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: pass
        config.update({"task": "multilabel", "label_names": list(LABEL_NAMES), "label_policy": audit["zero_meaning"],
                       "input_sha256": sha256(args.input), "features_sha256": sha256(args.features),
                       "versions": versions, "accepted_records": len(y), "training_rows": len(train), "test_rows": len(test),
                       "test_fraction_actual": len(test) / len(y), "audit": audit,
                       "split_method": "iterative multilabel stratification; unique sequences",
                       "decision_rule": "per-label probabilities >= threshold; no normalization",
                       "cnn_axis": "ESM mean-embedding coordinates, not residue positions"})
        write_json(output / "run_config.json", config)
        cv_results, test_results = [], []
        for name in args.models:
            trainer = importlib.import_module("multi_label_pipeline." + MODEL_REGISTRY[name]).train_predict
            cv_results.append(evaluate_cv(name, trainer, X[train], y[train], sequences[train], rows[train], splits, output, args))
            write_csv(output / "model_comparison_cv.csv", list(cv_results[0]), cv_results)
        for name in args.models:
            trainer = importlib.import_module("multi_label_pipeline." + MODEL_REGISTRY[name]).train_predict
            test_results.append(evaluate_holdout(name, trainer, X[train], y[train], sequences[train],
                                                 X[test], y[test], sequences[test], rows[test], output, args))
            write_csv(output / "model_comparison_test.csv", list(test_results[0]), test_results)
        from multi_label_pipeline.reporting import generate_reports
        generate_reports(output, args.models)
        for result in test_results:
            print(f"{result['model']}: exact-match={result['subset_accuracy']:.4f} "
                  f"macro-F1={result['f1_macro']:.4f} Hamming loss={result['hamming_loss']:.4f}")
        print(f"Results: {output}")
        return 0
    except (ImportError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr); return 1


if __name__ == "__main__":
    sys.exit(main())
