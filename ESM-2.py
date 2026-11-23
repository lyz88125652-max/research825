#!/usr/bin/env python3
"""Extract ESM-2 650M mean embeddings from CSV or XLSX.

Install: python -m pip install torch transformers numpy openpyxl
Run:     python ESM-2.py
Check:   python ESM-2.py --validate-only

Default input: data/multi-label.csv
Multi-label columns: sequence,antioxidant,antimicrobial,toxic
Legacy multi-class columns: sequence,label (labels 1, 2, 3)

Model: https://huggingface.co/facebook/esm2_t33_650M_UR50D
Method: https://github.com/facebookresearch/esm (mean, final layer)

Only uppercase ESM residue symbols are accepted. Modified notation and
lowercase residues are skipped and reported; no chemical conversion is inferred.
Duplicate sequences and their labels are retained in original input order.
No sequence truncation, normalization, scaling, or CV split is performed.
"""

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
MODEL_ID = "facebook/esm2_t33_650M_UR50D"
RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUO")
MAX_LENGTH = 1022
LABEL_NAMES = ("antioxidant", "antimicrobial", "toxic")
POOLING = "mean over residue tokens; exclude BOS, EOS and padding"


def sequence_error(sequence):
    if not isinstance(sequence, str) or not sequence:
        return "empty or non-text sequence"
    invalid = sorted(set(sequence) - RESIDUES)
    if invalid:
        return "unsupported characters: " + repr("".join(invalid))
    if len(sequence) > MAX_LENGTH:
        return f"length {len(sequence)} exceeds {MAX_LENGTH}; truncation disabled"
    return None


def read_dataset(path, task="auto"):
    """Return task and (source_row, original_sequence, target) records.

    A multi-label target is an ordered tuple of three binary integers.
    A legacy multi-class target is a scalar integer. No labels are inferred.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))
    elif path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            if len(workbook.sheetnames) != 1:
                raise ValueError("Input must have exactly one worksheet.")
            rows = list(workbook.active.iter_rows(values_only=True))
        finally:
            workbook.close()
    else:
        raise ValueError("Input must be a .csv or .xlsx file.")
    header = tuple(rows[0]) if rows else ()
    detected = {("sequence", "label"): "multiclass",
                ("sequence", *LABEL_NAMES): "multilabel"}.get(header)
    if detected is None:
        raise ValueError("Expected sequence,label or sequence,antioxidant,antimicrobial,toxic.")
    if task != "auto" and task != detected:
        raise ValueError(f"Requested {task}, but input header identifies {detected}.")
    accepted, skipped = [], []
    seen = set()
    for row_number, row in enumerate(rows[1:], start=2):
        if not row or all(value is None or value == "" for value in row):
            continue
        if len(row) != len(header):
            raise ValueError(f"Source row {row_number}: unexpected column count.")
        sequence = row[0]
        if detected == "multiclass":
            label = row[1]
            if isinstance(label, bool) or label not in (1, 2, 3, "1", "2", "3"):
                raise ValueError(f"Source row {row_number}: label must be 1, 2, or 3.")
            target = int(label)
        else:
            if any(isinstance(v, bool) or v not in (0, 1, "0", "1") for v in row[1:]):
                raise ValueError(f"Source row {row_number}: every label must be explicitly 0 or 1.")
            target = tuple(int(v) for v in row[1:])
            if isinstance(sequence, str):
                if sequence in seen:
                    raise ValueError(f"Source row {row_number}: duplicate sequence; merge labels first.")
                seen.add(sequence)
        record = (row_number, sequence, target)
        reason = sequence_error(sequence)
        if reason:
            skipped.append((*record, reason))
        else:
            accepted.append(record)
    return detected, accepted, skipped


def read_input(path):
    # Backward-compatible interface used by main_multi_class.py.
    _, accepted, skipped = read_dataset(path)
    return accepted, skipped


def write_skipped(path, records, task="multiclass"):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if task == "multilabel":
            writer.writerow(["source_row", "sequence", *LABEL_NAMES, "reason"])
            writer.writerows((row, sequence, *labels, reason) for row, sequence, labels, reason in records)
        else:
            writer.writerow(["excel_row", "sequence", "label", "reason"])
            writer.writerows(records)


def select_device(torch, requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Use --device cpu or auto.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available. Use --device cpu or auto.")
    return torch.device(requested)


def extract(records, args):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = select_device(torch, args.device)
    print(f"Model: {MODEL_ID}; device: {device}", flush=True)
    # The first run downloads weights from Hugging Face; subsequent runs use cache.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    if model.config.hidden_size != 1280 or model.config.num_hidden_layers != 33:
        raise ValueError("Unexpected model configuration; expected ESM-2 650M.")
    features = np.empty((len(records), 1280), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            sequences = [record[1] for record in batch]
            tokens = tokenizer(
                sequences, padding=True, truncation=False,
                return_tensors="pt", return_special_tokens_mask=True,
            )
            special_mask = tokens.pop("special_tokens_mask")
            residue_mask = tokens["attention_mask"].bool() & ~special_mask.bool()
            lengths = residue_mask.sum(dim=1).tolist()
            if lengths != [len(sequence) for sequence in sequences]:
                raise ValueError("Tokenizer residue counts do not match sequence lengths.")
            if (tokens["input_ids"] == tokenizer.unk_token_id).any().item():
                raise ValueError("Tokenizer produced an unknown residue token.")
            inputs = {key: value.to(device) for key, value in tokens.items()}
            # Final transformer layer; BOS, EOS, and padding are excluded.
            hidden = model(**inputs).last_hidden_state
            mask = residue_mask.to(device).unsqueeze(-1)
            pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1)
            values = pooled.cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite features in batch starting at {start}.")
            features[start:start + len(batch)] = values
            print(f"Extracted {start + len(batch)}/{len(records)}", flush=True)
    return features, str(device)


def encode_records(records, args):
    """Reuse matching sequence embeddings only; always rebuild targets from input."""
    import numpy as np

    features = np.empty((len(records), 1280), dtype=np.float32)
    reused, missing = 0, []
    lookup = {}
    if args.reuse_features:
        path = args.reuse_features
        metadata = json.loads(path.with_suffix(".metadata.json").read_text())
        if (metadata.get("model") != MODEL_ID or metadata.get("layer") != 33
                or metadata.get("features") != 1280 or metadata.get("pooling") != POOLING):
            raise ValueError("Reuse cache has incompatible model/layer/pooling metadata.")
        with np.load(path, allow_pickle=False) as cache:
            if not {"X", "sequences"}.issubset(cache.files):
                raise ValueError("Reuse cache requires X and sequences arrays.")
            X, sequences = cache["X"], cache["sequences"]
        if X.shape != (len(sequences), 1280) or not np.isfinite(X).all():
            raise ValueError("Invalid reuse cache features.")
        for sequence, vector in zip(sequences, X):
            key = str(sequence)
            if key in lookup and not np.allclose(vector, lookup[key], atol=1e-4, rtol=1e-3):
                raise ValueError(f"Inconsistent cached embeddings for sequence {key}.")
            lookup.setdefault(key, vector)
    for i, record in enumerate(records):
        if record[1] in lookup:
            features[i] = lookup[record[1]]
            reused += 1
        else:
            missing.append(i)
    device = "reused_cache"
    if missing:
        values, device = extract([records[i] for i in missing], args)
        features[missing] = values
    return features, device, reused


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data/multi-label.csv")
    parser.add_argument("--output", type=Path, help="NPZ output (default follows detected task).")
    parser.add_argument("--task", choices=["auto", "multilabel", "multiclass"], default="auto")
    parser.add_argument("--reuse-features", type=Path,
                        help="Reuse exact-sequence embeddings from a compatible NPZ plus metadata.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--invalid", choices=["skip", "error"], default="skip",
                        help="Report and skip unsupported sequences, or stop (default: skip).")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate CSV/XLSX without loading the model or extracting features.")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output is not None and args.output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    try:
        task, records, skipped = read_dataset(args.input, args.task)
        if args.output is None:
            args.output = ROOT / ("data/multi_label/multi_label_esm2_650M.npz" if task == "multilabel"
                                  else "data/multi_class/training_cv_esm2_650M.npz")
        if args.reuse_features and args.reuse_features.resolve() == args.output.resolve():
            raise ValueError("Reuse cache and output must be different files.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import hashlib
        import shutil
        from datetime import datetime
        input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
        backup_dir = None
        existing = [path for path in [args.output, args.output.with_suffix(".metadata.json"),
                    args.output.with_suffix(".skipped.csv")] if path.exists()]
        if existing and not args.validate_only:
            backup_dir = args.output.parent / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup_dir.mkdir(parents=True)
            for path in existing: shutil.copy2(path, backup_dir / path.name)
            print(f"Previous extraction backed up: {backup_dir}")
        skipped_path = args.output.with_suffix(".validation.skipped.csv" if args.validate_only else ".skipped.csv")
        write_skipped(skipped_path, skipped, task)
        counts = ({name: sum(r[2][i] for r in records) for i, name in enumerate(LABEL_NAMES)}
                  if task == "multilabel" else
                  {label: sum(r[2] == label for r in records) for label in (1, 2, 3)})
        print(f"Accepted: {len(records)}; skipped: {len(skipped)}; labels: {counts}")
        if skipped:
            print(f"Excluded sequences: {skipped_path}", file=sys.stderr)
        if skipped and args.invalid == "error":
            raise ValueError("Unsupported sequences found; see the exclusion report.")
        if not records:
            raise ValueError("No supported sequences to extract.")
        if args.validate_only:
            return 0
        import numpy as np

        features, device, reused = encode_records(records, args)
        if hashlib.sha256(args.input.read_bytes()).hexdigest() != input_hash:
            raise ValueError("Input changed during feature extraction.")
        # All arrays are aligned by row. Load with np.load(..., allow_pickle=False).
        np.savez_compressed(
            args.output, X=features,
            y=np.asarray([r[2] for r in records], dtype=np.int64),
            sequences=np.asarray([r[1] for r in records], dtype=str),
            excel_rows=np.asarray([r[0] for r in records], dtype=np.int64),
            source_rows=np.asarray([r[0] for r in records], dtype=np.int64),
            label_names=np.asarray(LABEL_NAMES, dtype=str),
            task=np.asarray(task),
        )
        metadata = {
            "model": MODEL_ID, "layer": 33, "features": 1280,
            "pooling": POOLING, "task": task,
            "input": str(args.input.resolve()), "output": str(args.output.resolve()),
            "accepted": len(records), "skipped": len(skipped), "label_counts": counts,
            "labels": ({1: "antioxidant", 2: "antimicrobial", 3: "toxic"}
                       if task == "multiclass" else list(LABEL_NAMES)),
            "label_names": list(LABEL_NAMES), "target_shape": [len(records), 3] if task == "multilabel" else [len(records)],
            "reused_records": reused,
            "reuse_features": str(args.reuse_features.resolve()) if args.reuse_features else None,
            "zero_label_meaning": "not recorded / negative or unknown" if task == "multilabel" else None,
            "invalid_policy": args.invalid, "device": device,
            "sequence_conversion": "none; lowercase and modification notation unsupported",
            "input_sha256": input_hash,
            "extraction_kind": "fresh_forward" if reused == 0 else "cache_reuse_or_mixed",
            "backup_directory": str(backup_dir.resolve()) if backup_dir else None,
            "created_at": datetime.now().astimezone().isoformat(),
            "python_version": sys.version,
        }
        import importlib.metadata
        metadata["versions"] = {}
        for package in ["numpy", "torch", "transformers"]:
            try: metadata["versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: pass
        args.output.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {args.output}; X={features.shape}; y={metadata['target_shape']}; reused={reused}")
        return 0
    except ImportError as exc:
        print(f"Missing dependency: {exc}.\nInstall: python -m pip install "
              "torch transformers numpy openpyxl", file=sys.stderr)
        return 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
