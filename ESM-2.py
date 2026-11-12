#!/usr/bin/env python3
"""Extract ESM-2 650M mean embeddings from data/training_cv.xlsx.

Install: python -m pip install torch transformers numpy openpyxl
Run:     python ESM-2.py
Check:   python ESM-2.py --validate-only

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


def sequence_error(sequence):
    if not isinstance(sequence, str) or not sequence:
        return "empty or non-text sequence"
    invalid = sorted(set(sequence) - RESIDUES)
    if invalid:
        return "unsupported characters: " + repr("".join(invalid))
    if len(sequence) > MAX_LENGTH:
        return f"length {len(sequence)} exceeds {MAX_LENGTH}; truncation disabled"
    return None


def read_input(path):
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    accepted, skipped = [], []
    try:
        if len(workbook.sheetnames) != 1:
            raise ValueError("Input must have exactly one worksheet.")
        rows = workbook.active.iter_rows(values_only=True)
        header = next(rows, ())
        if tuple(header) != ("sequence", "label"):
            raise ValueError("Expected exactly two columns: sequence, label.")
        for row_number, row in enumerate(rows, start=2):
            if all(value is None for value in row):
                continue
            sequence, label = row
            if isinstance(label, bool) or label not in (1, 2, 3):
                raise ValueError(f"Excel row {row_number}: label must be 1, 2, or 3.")
            record = (row_number, sequence, int(label))
            reason = sequence_error(sequence)
            if reason:
                skipped.append((*record, reason))
            else:
                accepted.append(record)
    finally:
        workbook.close()
    return accepted, skipped


def write_skipped(path, records):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data/training_cv.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "data/training_cv_esm2_650M.npz")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--invalid", choices=["skip", "error"], default="skip",
                        help="Report and skip unsupported sequences, or stop (default: skip).")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate the XLSX without loading the model or extracting features.")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    try:
        records, skipped = read_input(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        skipped_path = args.output.with_suffix(".skipped.csv")
        write_skipped(skipped_path, skipped)
        counts = {label: sum(r[2] == label for r in records) for label in (1, 2, 3)}
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

        features, device = extract(records, args)
        # All arrays are aligned by row. Load with np.load(..., allow_pickle=False).
        np.savez_compressed(
            args.output, X=features,
            y=np.asarray([r[2] for r in records], dtype=np.int64),
            sequences=np.asarray([r[1] for r in records], dtype=str),
            excel_rows=np.asarray([r[0] for r in records], dtype=np.int64),
        )
        metadata = {
            "model": MODEL_ID, "layer": 33, "features": 1280,
            "pooling": "mean over residue tokens; exclude BOS, EOS and padding",
            "input": str(args.input.resolve()), "output": str(args.output.resolve()),
            "accepted": len(records), "skipped": len(skipped), "label_counts": counts,
            "labels": {1: "antioxidant", 2: "antimicrobial", 3: "toxic"},
            "invalid_policy": args.invalid, "device": device,
            "sequence_conversion": "none; lowercase and modification notation unsupported",
        }
        args.output.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {args.output}; X={features.shape}; y=({len(records)},)")
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
