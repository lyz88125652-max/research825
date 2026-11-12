"""Run with: python -m unittest discover -s tests -v.

All learning tests use synthetic features, never scientific benchmark scores.
"""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from openpyxl import Workbook, load_workbook

from ml_pipeline.data import audit_labels, load_features, make_splits


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_group_disjoint_and_complete(self):
        y = np.repeat(np.arange(3), 24)
        groups = np.asarray([f"class{c}_seq{i // 2}" for c in range(3) for i in range(24)])
        seen = np.zeros(len(y), dtype=int)
        splits = make_splits(y, groups, 3, 42)
        for train, test in splits:
            self.assertFalse(set(groups[train]) & set(groups[test]))
            seen[test] += 1
        np.testing.assert_array_equal(seen, np.ones(len(y)))
        for (a, b), (c, d) in zip(splits, make_splits(y, groups, 3, 42)):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)

    def test_conflicts_removed_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            y = np.asarray([0, 1, 2, 0, 0])
            sequences = np.asarray(["AAA", "AAA", "CCC", "DDD", "DDD"])
            rows = np.arange(2, 7)
            keep, stats = audit_labels(y, sequences, rows, output, "drop")
            np.testing.assert_array_equal(keep, [False, False, True, True, True])
            self.assertEqual(stats["conflicting_sequences"], 1)
            self.assertEqual(stats["excluded_conflicting_rows"], 2)
            workbook = load_workbook(output / "duplicated.xlsx", read_only=True)
            values = list(workbook.active.values)
            workbook.close()
            self.assertEqual(values[1:], [("AAA", 1, 2, "1;2"), ("AAA", 2, 3, "1;2")])
            keep, _ = audit_labels(y, sequences, rows, output, "keep")
            self.assertTrue(keep.all())

    def test_features_reject_nan_and_misalignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            X = np.ones((3, 1280), dtype=np.float32)
            data = dict(X=X, y=np.asarray([1, 2, 3]), sequences=np.asarray(["AAA", "CCC", "DDD"]),
                        excel_rows=np.asarray([2, 3, 4]))
            np.savez(path, **data)
            self.assertEqual(load_features(path)[0].shape, (3, 1280))
            class FakeExtractor:
                @staticmethod
                def read_input(_):
                    return [(2, "AAA", 1), (3, "CCC", 2), (4, "EEE", 3)], []
            with self.assertRaisesRegex(ValueError, "do not match"):
                load_features(path, Path("unused.xlsx"), FakeExtractor())
            X[0, 0] = np.nan
            np.savez(path, **data)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_features(path)

    def test_all_models_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rng = np.random.default_rng(42)
            y = np.repeat([1, 2, 3], 18)
            letters = "ACDEFGHIKLMNPQRSTVWY"
            sequences = ["AC" + letters[i // 20] + letters[i % 20] for i in range(len(y))]
            X = rng.normal(size=(len(y), 1280)).astype(np.float32)
            for label in (1, 2, 3):
                X[y == label, (label - 1) * 128:label * 128] += 2
            # A cross-class duplicate must be reported and both rows removed.
            sequences[-1] = sequences[0]
            X[-1] = X[0]
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["sequence", "label"])
            for sequence, label in zip(sequences, y):
                sheet.append([sequence, int(label)])
            input_path = directory / "training.xlsx"
            workbook.save(input_path)
            workbook.close()
            features = directory / "features.npz"
            np.savez(features, X=X, y=y, sequences=np.asarray(sequences),
                     excel_rows=np.arange(2, len(y) + 2))
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1")
            run = subprocess.run(
                [sys.executable, str(ROOT / "main.py"), "--input", str(input_path),
                 "--features", str(features), "--output-dir", str(directory / "results"),
                 "--folds", "3", "--trees", "3", "--jobs", "1", "--cnn-epochs", "1",
                 "--cnn-batch-size", "16", "--device", "cpu"],
                cwd=ROOT, env=env, text=True, capture_output=True, timeout=240,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            output = next((directory / "results").glob("run_*"))
            audit = json.loads((output / "data_audit.json").read_text())
            self.assertEqual(audit["conflicting_sequences"], 1)
            self.assertEqual(audit["eligible_training_rows"], 52)
            with (output / "model_comparison.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)
            reference_folds = None
            for name in ("cnn", "rf", "lightgbm"):
                folder = output / name
                with (folder / "predictions.csv").open() as handle:
                    predictions = list(csv.DictReader(handle))
                self.assertEqual(len(predictions), 52)
                self.assertNotIn(sequences[0], [r["sequence"] for r in predictions])
                assignments = [(r["excel_row"], r["fold"]) for r in predictions]
                if reference_folds is None:
                    reference_folds = assignments
                self.assertEqual(assignments, reference_folds)
                config = json.loads((folder / "summary.json").read_text())
                self.assertTrue(0 <= config["pooled_out_of_fold"]["accuracy"] <= 1)
                for fold in range(1, 4):
                    self.assertTrue(list((folder / f"fold_{fold}").glob("model.*")))
                fold_predictions = [r for r in predictions if r["fold"] == "1"]
                indices = [int(r["excel_row"]) - 2 for r in fold_predictions]
                expected = np.asarray([[float(r[f"prob_label_{c}"]) for c in (1, 2, 3)]
                                       for r in fold_predictions])
                if name == "cnn":
                    import torch
                    from ml_pipeline.cnn import build_network
                    checkpoint = torch.load(folder / "fold_1/model.pt", weights_only=True)
                    model = build_network()
                    model.load_state_dict(checkpoint["state_dict"])
                    self.assertEqual(checkpoint["input_dim"], 1280)
                    standardized = ((X[indices] - checkpoint["scaler_mean"].numpy())
                                    / checkpoint["scaler_scale"].numpy()).astype(np.float32)
                    model.eval()
                    with torch.inference_mode():
                        actual = torch.softmax(model(torch.from_numpy(standardized).unsqueeze(1)),
                                               dim=1).numpy()
                else:
                    import joblib
                    model = joblib.load(folder / "fold_1/model.joblib")
                    actual = model.predict_proba(X[indices])
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
