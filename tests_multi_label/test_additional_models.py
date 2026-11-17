"""Contract checks without ESM extraction or classifier fitting.

python -m unittest discover -s tests_multi_label -p test_additional_models.py -v
"""

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
import tempfile
import sys

import numpy as np

from main_multi_label import MODEL_REGISTRY, parse_args
from multi_label_pipeline.model_utils import BinaryRelevancePredictor, positive_probabilities


class AdditionalModelTests(TestCase):
    def test_registry_and_cli(self):
        args = parse_args([])
        self.assertEqual(len(args.models), 9)
        self.assertEqual(args.chain_order, [0, 1, 2])
        for module in MODEL_REGISTRY.values():
            self.assertTrue(callable(importlib.import_module("multi_label_pipeline." + module).train_predict))
        self.assertEqual(parse_args(["--models", "svm", "lr", "svm"]).models, ["svm", "lr"])
        for flags in [["--chain-order", "0", "0", "2"], ["--linear-c", "nan"], ["--mlp-epochs", "0"]]:
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit): parse_args(flags)

    def test_positive_class_and_label_order(self):
        first = SimpleNamespace(classes_=np.array([0, 1]), predict_proba=lambda X: np.array([[.1, .9], [.8, .2]]))
        second = SimpleNamespace(classes_=np.array([1, 0]), predict_proba=lambda X: np.array([[.8, .2], [.4, .6]]))
        model = BinaryRelevancePredictor([first, second, first])
        actual = positive_probabilities(model, np.zeros((2, 1280)))
        np.testing.assert_allclose(actual, [[.9, .8, .9], [.2, .4, .2]])
        self.assertFalse(np.allclose(actual.sum(1), 1))

    def test_mlp_forward_and_shared_training_contract(self):
        import torch
        from multi_label_pipeline import mlp
        network = mlp.build_network().eval()
        with torch.inference_mode():
            p = torch.sigmoid(network(torch.zeros((2, 1, 1280))))
        self.assertEqual(tuple(p.shape), (2, 3))
        self.assertTrue(bool(((p >= 0) & (p <= 1)).all()))
        args = parse_args([])
        with mock.patch("multi_label_pipeline.cnn.train_predict", return_value=("prob", {})) as trainer:
            self.assertEqual(mlp.train_predict(None, None, None, None, None, args, 42)[0], "prob")
            self.assertIs(trainer.call_args.kwargs["network_factory"], mlp.build_network)
            self.assertEqual(trainer.call_args.kwargs["epochs"], args.mlp_epochs)

    def test_xgboost_uses_training_counts_without_fitting(self):
        from multi_label_pipeline.xgboost_model import train_predict
        args = parse_args([])
        y = np.array([[1, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, 1]])
        estimators = []
        def create(**params):
            estimator = mock.Mock(classes_=np.array([0, 1]))
            estimator.predict_proba.return_value = np.array([[.2, .8], [.3, .7]])
            estimators.append((params, estimator))
            return estimator
        fake = SimpleNamespace(XGBClassifier=create)
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(sys.modules, {"xgboost": fake}), mock.patch("joblib.dump"):
            p, details = train_predict(np.zeros((4, 1280)), y, np.zeros((2, 1280)), None, Path(folder), args, 42)
        self.assertEqual(p.shape, (2, 3))
        np.testing.assert_allclose(details["scale_pos_weight"], [3, 1/3, 1/3])
        for index, (params, estimator) in enumerate(estimators):
            self.assertEqual(params["objective"], "binary:logistic")
            np.testing.assert_array_equal(estimator.fit.call_args.args[1], y[:, index])

    def test_catboost_native_multilabel_without_fitting(self):
        from multi_label_pipeline.catboost_model import train_predict
        model = mock.Mock()
        expected = np.array([[.8, .9, .6]])
        model.predict_proba.return_value = expected
        factory = mock.Mock(return_value=model)
        y = np.array([[1, 0, 1], [0, 1, 1]])
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(sys.modules, {"catboost": SimpleNamespace(CatBoostClassifier=factory)}), mock.patch("joblib.dump"):
            p, _ = train_predict(np.zeros((2, 1280)), y, np.zeros((1, 1280)), None, Path(folder), parse_args([]), 42)
        self.assertEqual(factory.call_args.kwargs["loss_function"], "MultiLogloss")
        np.testing.assert_array_equal(model.fit.call_args.args[1], y)
        np.testing.assert_array_equal(p, expected)

    def test_chain_internal_cv_and_fixed_order_without_fitting(self):
        from multi_label_pipeline.classifier_chain import train_predict
        y = np.tile([[1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]], (6, 1))
        model = mock.Mock()
        model.predict_proba.return_value = np.array([[.6, .7, .8]])
        factory = mock.Mock(return_value=model)
        args = parse_args(["--chain-order", "2", "0", "1"])
        with tempfile.TemporaryDirectory() as folder, mock.patch("sklearn.multioutput.ClassifierChain", factory), mock.patch("joblib.dump"):
            p, details = train_predict(np.zeros((24, 1280)), y, np.zeros((1, 1280)),
                                       np.array([f"sequence{i}" for i in range(24)]), Path(folder), args, 42)
        self.assertEqual(factory.call_args.kwargs["order"], [2, 0, 1])
        splits = factory.call_args.kwargs["cv"]
        self.assertEqual(len(splits), 3)
        seen = np.zeros(24, dtype=int)
        for fit, validation in splits:
            self.assertFalse(set(fit) & set(validation)); seen[validation] += 1
        np.testing.assert_array_equal(seen, np.ones(24))
        self.assertEqual(details["order"], ["toxic", "antioxidant", "antimicrobial"])
        self.assertEqual(p.shape, (1, 3))

    def test_lr_and_svm_scaling_and_probability_contract_without_fitting(self):
        for name, estimator_path in [("logistic_regression", "sklearn.linear_model.LogisticRegression"),
                                      ("svm", "sklearn.svm.SVC")]:
            trainer = importlib.import_module("multi_label_pipeline." + name).train_predict
            pipeline = mock.Mock()
            classifiers = mock.Mock(estimators_=[SimpleNamespace(classes_=[0, 1])] * 3)
            classifiers.predict_proba.return_value = [np.array([[.2, .8]])] * 3
            pipeline.named_steps = {"scaler": mock.Mock(), "classifiers": classifiers}
            X, y = np.zeros((8, 1280)), np.tile([[1, 0, 1], [0, 1, 0]], (4, 1))
            with tempfile.TemporaryDirectory() as folder, mock.patch("sklearn.pipeline.Pipeline", return_value=pipeline) as constructor, mock.patch(estimator_path) as estimator, mock.patch("joblib.dump"):
                p, _ = trainer(X, y, X[:1], None, Path(folder), parse_args([]), 42)
            self.assertEqual(p.shape, (1, 3))
            self.assertEqual([step[0] for step in constructor.call_args.args[0]], ["scaler", "classifiers"])
            pipeline.fit.assert_called_once()
            np.testing.assert_array_equal(pipeline.fit.call_args.args[0], X)
            self.assertEqual(estimator.call_args.kwargs["class_weight"], "balanced")
            if name == "svm": self.assertTrue(estimator.call_args.kwargs["probability"])

    def test_nine_model_reports_without_training(self):
        from multi_label_pipeline.evaluation import save_predictions, write_csv, write_json
        from multi_label_pipeline.reporting import generate_reports
        from openpyxl import load_workbook
        y = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        p = .1 + .8 * y
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            for name in MODEL_REGISTRY:
                target = output / name; target.mkdir()
                metrics, _ = save_predictions(target, "test_", y, p, ["A", "C", "D", "E"], [2, 3, 4, 5], .5)
                save_predictions(target, "oof_", y, p, ["F", "G", "H", "I"], [6, 7, 8, 9], .5, [1, 1, 2, 2])
                means = {key + suffix: value if suffix == "_mean" else 0.0
                         for key, value in metrics.items() for suffix in ["_mean", "_std"]}
                write_json(target / "summary.json", {"cv_fold_mean_std":means, "pooled_out_of_fold":metrics})
                row = {"fold":1, "train_size":4, "validation_size":4, **metrics}
                write_csv(target / "fold_metrics.csv", list(row), [row])
                if name in {"cnn", "mlp"}:
                    for part in ["fold_1", "final_model"]:
                        destination = target / part; destination.mkdir()
                        write_csv(destination / "history.csv", ["phase", "epoch", "train_eval_loss", "validation_loss"],
                                  [{"phase":"selection", "epoch":1, "train_eval_loss":.3, "validation_loss":.4}])
                        write_json(destination / "metrics.json", {"training":{"selected_epochs":1}})
            generate_reports(output, list(MODEL_REGISTRY))
            for filename in ["summary.xlsx", "model_comparison.png", "test_roc.png", "test_pr.png",
                             "cnn/loss_curves.png", "mlp/loss_curves.png"]:
                self.assertTrue((output / filename).exists(), filename)
            workbook = load_workbook(output / "summary.xlsx", read_only=True)
            self.assertEqual(workbook["Summary"].max_row, 1 + 9 * 4)
            workbook.close()
