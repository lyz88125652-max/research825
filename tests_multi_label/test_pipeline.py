import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from openpyxl import load_workbook

from multi_label_pipeline.data import make_holdout, make_splits
from multi_label_pipeline.metrics import score, binary_log_loss

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["antioxidant", "antimicrobial", "toxic"]


def dataset():
    patterns = [[1, 0, 0]] * 20 + [[0, 1, 0]] * 20 + [[0, 0, 1]] * 20 + [[0, 1, 1]] * 20
    patterns += [[1, 1, 0]] * 3 + [[1, 1, 1]] * 2
    y = np.asarray(patterns)
    letters = "ACDEFGHIKLMNPQRSTVWY"
    sequences = np.asarray(["AC" + letters[i // 20] + letters[i % 20] for i in range(len(y))])
    return y, sequences


class PipelineTests(unittest.TestCase):
    def test_iterative_split_keeps_rare_combinations(self):
        y, sequences = dataset()
        train, test = make_holdout(y, sequences, 42)
        self.assertEqual(len(train) + len(test), len(y))
        self.assertFalse(set(sequences[train]) & set(sequences[test]))
        self.assertEqual(int((y.sum(1) == 3).sum()), 2)
        np.testing.assert_array_equal(make_holdout(y, sequences, 42)[0], train)
        seen = np.zeros(len(train), dtype=int)
        for fit, validation in make_splits(y[train], sequences[train], 3, 42):
            self.assertFalse(set(sequences[train][fit]) & set(sequences[train][validation]))
            self.assertFalse(set(sequences[train][fit]) & set(sequences[test]))
            seen[validation] += 1
        np.testing.assert_array_equal(seen, np.ones(len(train)))

    def test_metrics_use_independent_probabilities(self):
        y = np.asarray([[1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1]])
        p = np.asarray([[.9, .9, .2], [.2, .9, .8], [.6, .1, .1], [.7, .2, .8]])
        overall, classes = score(y, p)
        self.assertAlmostEqual(overall["subset_accuracy"], .75)
        self.assertAlmostEqual(overall["hamming_loss"], 1 / 12)
        self.assertAlmostEqual(classes[0]["sensitivity"], 1)
        self.assertAlmostEqual(classes[0]["specificity"], .5)
        self.assertEqual(classes[0]["tp"], 2)
        self.assertEqual(classes[0]["fp"], 1)
        self.assertTrue(np.isfinite(binary_log_loss(y, np.ones(y.shape, dtype=np.float32))))
        _, absent = score(np.zeros((4, 3), dtype=int), np.full((4, 3), .1))
        self.assertIsNone(absent[0]["roc_auc"])
        self.assertIsNone(absent[0]["sensitivity"])

    def test_all_three_models_end_to_end(self):
        import joblib
        import torch
        from multi_label_pipeline.cnn import build_network
        y, sequences = dataset()
        rng = np.random.default_rng(42)
        X = rng.normal(size=(len(y), 1280)).astype(np.float32)
        for c in range(3): X[:, c * 128:(c + 1) * 128] += 2 * y[:, c:c + 1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, features = directory / "labels.csv", directory / "features.npz"
            with source.open("w", newline="") as handle:
                writer = csv.writer(handle); writer.writerow(["sequence", *NAMES])
                writer.writerows((sequence, *label) for sequence, label in zip(sequences, y))
            np.savez(features, X=X, y=y, sequences=sequences, source_rows=np.arange(2, len(y) + 2),
                     label_names=np.asarray(NAMES), task=np.asarray("multilabel"))
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            run = subprocess.run([sys.executable, str(ROOT / "main_multi_label.py"),
                                  "--models", "cnn", "rf", "lightgbm",
                                  "--input", str(source), "--features", str(features),
                                  "--output-dir", str(directory / "results"), "--folds", "3", "--trees", "3",
                                  "--jobs", "1", "--cnn-epochs", "1", "--cnn-batch-size", "16", "--device", "cpu"],
                                 cwd=ROOT, env=env, text=True, capture_output=True, timeout=240)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            output = next((directory / "results").glob("run_*"))
            config = json.loads((output / "run_config.json").read_text())
            self.assertEqual(config["training_rows"] + config["test_rows"], len(y))
            self.assertEqual(config["audit"]["retained_multi_activity_records"], 25)
            workbook = load_workbook(output / "summary.xlsx", read_only=True)
            self.assertEqual(workbook.sheetnames, ["Summary", "CV_folds", "Label_metrics", "Notes"])
            workbook.close()
            reference = None
            for name in ["cnn", "rf", "lightgbm"]:
                folder = output / name
                with (folder / "test_predictions.csv").open(newline="") as handle:
                    test = list(csv.DictReader(handle))
                with (folder / "oof_predictions.csv").open(newline="") as handle:
                    oof = list(csv.DictReader(handle))
                test_sequences = {r["sequence"] for r in test}
                self.assertFalse(test_sequences & {r["sequence"] for r in oof})
                if reference is None: reference = [r["source_row"] for r in test]
                self.assertEqual(reference, [r["source_row"] for r in test])
                indices = [int(r["source_row"]) - 2 for r in test]
                expected = np.asarray([[float(r["prob_" + c]) for c in NAMES] for r in test])
                self.assertEqual(expected.shape, (config["test_rows"], 3))
                self.assertFalse(np.allclose(expected.sum(1), 1))
                if name == "cnn":
                    checkpoint = torch.load(folder / "final_model/model.pt", weights_only=True)
                    model = build_network(); model.load_state_dict(checkpoint["state_dict"]); model.eval()
                    values = ((X[indices] - checkpoint["scaler_mean"].numpy())
                              / checkpoint["scaler_scale"].numpy()).astype(np.float32)
                    with torch.inference_mode():
                        actual = torch.sigmoid(model(torch.from_numpy(values).unsqueeze(1))).numpy()
                else:
                    model = joblib.load(folder / "final_model/model.joblib")
                    actual = np.column_stack([p[:, list(est.classes_).index(1)]
                                              for est, p in zip(model.estimators_, model.predict_proba(X[indices]))])
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
                saved = json.loads((folder / "test_metrics.json").read_text())
                recomputed, _ = score(y[indices], expected)
                self.assertAlmostEqual(saved["f1_macro"], recomputed["f1_macro"])
                self.assertTrue((folder / "test_confusion_matrices.png").exists())
            for file in ["test_roc.png", "test_pr.png", "model_comparison.png", "cnn/loss_curves.png"]:
                self.assertTrue((output / file).exists())


if __name__ == "__main__": unittest.main()
