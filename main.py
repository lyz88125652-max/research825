#!/usr/bin/env python3
"""Run ESM-2 extraction (if needed) and CNN / Random Forest / LightGBM CV.

    python main.py
    python main.py --models rf lightgbm
    python main.py --models cnn --device cpu --cnn-epochs 10
    python main.py --conflicts drop
"""

import argparse
from datetime import datetime
import hashlib
import importlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data/training_cv.xlsx")
    parser.add_argument("--features", type=Path, default=ROOT / "data/training_cv_esm2_650M.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--models", nargs="+", choices=["cnn", "rf", "lightgbm"],
                        default=["cnn", "rf", "lightgbm"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conflicts", choices=["keep", "drop"], default="drop",
                        help="Keep original labels or exclude sequences with multiple labels.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--esm-batch-size", type=int, default=8)
    parser.add_argument("--reextract", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Export duplicated.xlsx and audit counts without loading models.")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--cnn-epochs", type=int, default=30)
    parser.add_argument("--cnn-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args(argv)
    for option in ("folds", "esm_batch_size", "trees", "cnn_epochs", "cnn_batch_size", "patience"):
        if getattr(args, option) < (2 if option == "folds" else 1):
            parser.error(f"Invalid --{option.replace('_', '-')}")
    if args.learning_rate <= 0 or args.jobs == 0 or args.seed < 0:
        parser.error("Learning rate must be positive; jobs nonzero; seed nonnegative.")
    if args.features.suffix != ".npz":
        parser.error("--features must end in .npz")
    args.models = list(dict.fromkeys(args.models))
    return args


def check_dependencies(args):
    modules = {"numpy", "openpyxl"}
    if args.prepare_only:
        for name in sorted(modules):
            importlib.import_module(name)
        return
    modules.update({"sklearn", "joblib"})
    if "cnn" in args.models or args.reextract or not args.features.exists():
        modules.add("torch")
    if args.reextract or not args.features.exists():
        modules.add("transformers")
    if "lightgbm" in args.models:
        modules.add("lightgbm")
    failures = []
    for name in sorted(modules):
        try:
            importlib.import_module(name)
        except (ImportError, OSError) as exc:
            failures.append(f"{name}: {exc}")
    if failures:
        raise RuntimeError("Dependencies unavailable:\n" + "\n".join(failures)
                           + f"\nInstall: {sys.executable} -m pip install -r "
                           + str(ROOT / "requirements.txt")
                           + "\nOn macOS, a LightGBM libomp error can be resolved with: brew install libomp")


def main(argv=None):
    args = parse_args(argv)
    try:
        check_dependencies(args)
        from ml_pipeline.data import load_extractor, load_features, audit_labels, make_splits
        from ml_pipeline.evaluation import evaluate_model, write_csv, write_json
        import numpy as np

        output = args.output_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True)
        extractor = load_extractor(ROOT)
        accepted, skipped = extractor.read_input(args.input)
        original = sorted(accepted + [record[:3] for record in skipped])
        all_rows = np.asarray([r[0] for r in original], dtype=np.int64)
        all_sequences = np.asarray([r[1] for r in original], dtype=object)
        all_y = np.asarray([r[2] - 1 for r in original], dtype=np.int64)
        keep, audit = audit_labels(all_y, all_sequences, all_rows, output, args.conflicts)
        kept_rows = set(all_rows[keep].tolist())
        audit["unsupported_sequence_rows"] = len(skipped)
        audit["eligible_training_rows"] = sum(r[0] in kept_rows for r in accepted)
        write_json(output / "data_audit.json", audit)
        if audit["conflicting_sequences"]:
            print(f"Found {audit['conflicting_sequences']} sequences with multiple labels; "
                  f"policy={args.conflicts}. See {output / 'duplicated.xlsx'}.", flush=True)
        if args.prepare_only:
            print(f"Eligible rows: {audit['eligible_training_rows']}; audit saved: {output}")
            return 0
        if args.reextract or not args.features.exists():
            print("Extracting ESM-2 650M embeddings before training...", flush=True)
            subprocess.run([sys.executable, str(ROOT / "ESM-2.py"),
                            "--input", str(args.input), "--output", str(args.features),
                            "--batch-size", str(args.esm_batch_size), "--device", args.device],
                           check=True)
        X, y, sequences, rows = load_features(args.features, args.input, extractor)
        keep = np.asarray([int(row) in kept_rows for row in rows])
        X, y, sequences, rows = X[keep], y[keep], sequences[keep], rows[keep]
        splits = make_splits(y, sequences, args.folds, args.seed)
        fold_ids = {}
        for fold, (_, test) in enumerate(splits, start=1):
            fold_ids.update({int(index): fold for index in test})
        write_csv(output / "fold_assignments.csv", ["excel_row", "sequence", "label", "fold"],
                  ({"excel_row": int(rows[i]), "sequence": str(sequences[i]),
                    "label": int(y[i]) + 1, "fold": fold_ids[i]} for i in range(len(y))))
        versions = {}
        for package in ("numpy", "scikit-learn", "torch", "lightgbm", "transformers", "openpyxl"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        digest = hashlib.sha256()
        with args.features.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
        config.update({"audit": audit, "training_rows": len(y), "feature_dimensions": X.shape[1],
                       "feature_sha256": digest.hexdigest(), "versions": versions,
                       "task": "multiclass; labels describe source annotations",
                       "split": "StratifiedGroupKFold grouped by exact sequence",
                       "labels": {1: "antioxidant", 2: "antimicrobial", 3: "toxic"},
                       "cnn_input": "pooled ESM embedding coordinates, not residue positions"})
        write_json(output / "run_config.json", config)
        print(f"Training rows: {len(y)}; features: {X.shape[1]}; results: {output}", flush=True)
        registry = {"cnn": "ml_pipeline.cnn", "rf": "ml_pipeline.random_forest",
                    "lightgbm": "ml_pipeline.lightgbm_model"}
        summaries = []
        for name in args.models:
            trainer = importlib.import_module(registry[name]).train_predict
            summaries.append(evaluate_model(name, trainer, X, y, sequences, rows,
                                            splits, output, args))
            write_csv(output / "model_comparison.csv", list(summaries[0]), summaries)
        print("\nCV results (mean ± sample standard deviation):", flush=True)
        for result in summaries:
            print(f"  {result['model']:10s} accuracy={result['accuracy_mean']:.4f} "
                  f"± {result['accuracy_std']:.4f}; macro-F1={result['f1_macro_mean']:.4f} "
                  f"± {result['f1_macro_std']:.4f}")
        print(f"Saved results: {output}")
        return 0
    except (ImportError, ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
