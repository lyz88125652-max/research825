import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from openpyxl import Workbook

from multi_label_pipeline.data import load_features

ROOT = Path(__file__).resolve().parents[1]


def extractor():
    spec = importlib.util.spec_from_file_location("extractor_under_test", ROOT / "ESM-2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_input(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "antioxidant", "antimicrobial", "toxic"])
        writer.writerows(records)


class ExtractorTests(unittest.TestCase):
    def test_vectors_survive_extraction_and_skip_report(self):
        module = extractor()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, output = directory / "labels.csv", directory / "features.npz"
            write_input(source, [("AAA", 1, 1, 0), ("CCC", 0, 0, 1), ("{SeM}PGP", 1, 0, 0)])
            def fake_extract(records, args):
                return np.ones((len(records), 1280), dtype=np.float32), "test_cpu"
            with patch.object(module, "extract", side_effect=fake_extract):
                self.assertEqual(module.main(["--input", str(source), "--output", str(output)]), 0)
            with np.load(output, allow_pickle=False) as result:
                np.testing.assert_array_equal(result["y"], [[1, 1, 0], [0, 0, 1]])
                np.testing.assert_array_equal(result["source_rows"], [2, 3])
                self.assertEqual(result["task"].item(), "multilabel")
            _, records, skipped = module.read_dataset(source)
            self.assertEqual(load_features(output, records)[0].shape, (2, 1280))
            with output.with_suffix(".skipped.csv").open(newline="") as handle:
                report = list(csv.DictReader(handle))
            self.assertEqual(report[0]["antioxidant"], "1")
            self.assertEqual(report[0]["sequence"], "{SeM}PGP")
            changed = [(records[0][0], records[0][1], (0, 1, 0)), records[1]]
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_features(output, changed)

    def test_duplicate_missing_and_nonbinary_labels_rejected(self):
        module = extractor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            for records, message in [([("AAA", 1, 0, 0), ("AAA", 0, 1, 0)], "duplicate"),
                                     ([("AAA", 1, "", 0)], "explicitly"),
                                     ([("AAA", 2, 0, 0)], "explicitly")]:
                write_input(path, records)
                with self.assertRaisesRegex(ValueError, message): module.read_dataset(path)

    def test_legacy_xlsx_remains_scalar(self):
        module = extractor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.xlsx"
            workbook = Workbook(); worksheet = workbook.active
            worksheet.append(["sequence", "label"])
            worksheet.append(["AAA", 1]); worksheet.append(["CCC", 3])
            workbook.save(path); workbook.close()
            task, records, skipped = module.read_dataset(path)
            self.assertEqual(task, "multiclass")
            self.assertEqual(records, [(2, "AAA", 1), (3, "CCC", 3)])
            self.assertEqual(module.read_input(path), (records, skipped))

    def test_legacy_cli_writes_one_dimensional_targets(self):
        module = extractor()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, output = directory / "legacy.csv", directory / "legacy.npz"
            source.write_text("sequence,label\nAAA,1\nCCC,3\n")
            with patch.object(module, "extract", return_value=(np.ones((2, 1280), dtype=np.float32), "test_cpu")):
                self.assertEqual(module.main(["--input", str(source), "--output", str(output)]), 0)
            with np.load(output, allow_pickle=False) as result:
                np.testing.assert_array_equal(result["y"], [1, 3])
                np.testing.assert_array_equal(result["excel_rows"], [2, 3])
                self.assertEqual(result["task"].item(), "multiclass")

    def test_reuse_rebuilds_targets_without_loading_model(self):
        module = extractor()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, output, cache = directory / "labels.csv", directory / "new.npz", directory / "old.npz"
            write_input(source, [("AAA", 1, 1, 0), ("CCC", 0, 0, 1)])
            vectors = np.arange(2 * 1280, dtype=np.float32).reshape(2, 1280)
            np.savez(cache, X=vectors, sequences=np.asarray(["AAA", "CCC"]), y=np.asarray([1, 2]))
            cache.with_suffix(".metadata.json").write_text(json.dumps({
                "model": module.MODEL_ID, "layer": 33, "features": 1280, "pooling": module.POOLING}))
            with patch.object(module, "extract", side_effect=AssertionError("Model should not load")):
                self.assertEqual(module.main(["--input", str(source), "--output", str(output),
                                               "--reuse-features", str(cache)]), 0)
            with np.load(output, allow_pickle=False) as result:
                np.testing.assert_array_equal(result["X"], vectors)
                np.testing.assert_array_equal(result["y"], [[1, 1, 0], [0, 0, 1]])


if __name__ == "__main__": unittest.main()
