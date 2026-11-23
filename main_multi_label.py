#!/usr/bin/env python3
"""Stage-separated multi-label experiments; final holdout is accessed explicitly.

python main_multi_label.py
python main_multi_label.py --prepare-only
python main_multi_label.py --models rf lightgbm
python main_multi_label.py --stage ensemble --run-dir results_multi_label/run_...
python main_multi_label.py --stage final --run-dir results_multi_label/run_...
"""

import argparse
import importlib
from pathlib import Path
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
    p.add_argument("--stage", choices=["prepare", "cv", "ensemble", "nested", "bias", "select", "final", "all"], default="cv")
    p.add_argument("--run-dir", type=Path, help="Resume a fixed experiment. Use only this option and --stage.")
    p.add_argument("--split-file", type=Path, help="Create once or reload frozen train/test and outer-CV assignments.")
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--meta-c", type=float, default=1.0)
    p.add_argument("--meta-class-weight", choices=["none", "balanced"], default="none")
    p.add_argument("--length-range", type=int, nargs=2, help="Optional prespecified common length range for training OOF analysis.")
    p.add_argument("--allow-reused-features", action="store_true", help="Exploratory run only: allow old cached embeddings.")
    p.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY), default=list(MODEL_REGISTRY))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--esm-batch-size", type=int, default=8, help=argparse.SUPPRESS)
    p.add_argument("--reextract", action="store_true", help="Legacy option: use ESM-2.py instead.")
    p.add_argument("--reuse-features", type=Path, help="Legacy option: use ESM-2.py instead.")
    p.add_argument("--no-feature-reuse", action="store_true", help="Legacy option: use ESM-2.py instead.")
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
                   "mlp_epochs", "mlp_batch_size", "chain_folds", "inner_folds", "top_k"]:
        if getattr(args, option) < (2 if option in {"folds", "chain_folds", "inner_folds"} else 1): p.error(f"Invalid {option}.")
    if not 0 < args.test_fraction < 0.5 or not 0 < args.threshold < 1:
        p.error("Use 0 < test-fraction < 0.5 and 0 < threshold < 1.")
    if args.jobs == 0 or args.learning_rate <= 0 or args.seed < 0:
        p.error("jobs must be nonzero; learning-rate positive; seed nonnegative.")
    if args.features.suffix.lower() != ".npz": p.error("Features must be an .npz file.")
    if args.no_feature_reuse and args.reuse_features: p.error("Conflicting feature-reuse options.")
    import math
    if any(not math.isfinite(value) or value <= 0 for value in [args.linear_c, args.svm_c, args.meta_c, args.learning_rate]):
        p.error("C and learning-rate values must be finite and positive.")
    if sorted(args.chain_order) != [0, 1, 2]: p.error("chain-order must be a permutation of 0 1 2.")
    args.models = list(dict.fromkeys(args.models))
    if args.length_range and not 0 < args.length_range[0] <= args.length_range[1]: p.error("Invalid length range.")
    if args.prepare_only: args.stage = "prepare"
    if args.run_dir:
        raw = list(sys.argv[1:] if argv is None else argv)
        allowed = {"--stage", "--run-dir"}
        if any(value.startswith("--") and value.split("=",1)[0] not in allowed for value in raw):
            p.error("When resuming, use only --run-dir and --stage; settings come from run_config.json.")
    return args


def dependency_check(args):
    modules = ["numpy", "sklearn", "iterstrat"]
    if args.stage != "prepare":
        modules += ["openpyxl", "matplotlib"]
        if args.stage in {"cv", "nested", "all", "final"}:
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


def main(argv=None):
    args = parse_args(argv)
    try:
        from multi_label_pipeline.workflow import run, restore_args
        if args.run_dir:
            restored, _ = restore_args(args)
            dependency_check(restored)
        else:
            dependency_check(args)
        output = run(args, ROOT, MODEL_REGISTRY)
        print(f"Completed: {output}")
        return 0
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
