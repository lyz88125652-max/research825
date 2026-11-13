"""Export an Excel summary and CNN loss plots from saved run artifacts.

Usage: python -m ml_pipeline.reporting results/run_YYYYMMDD_HHMMSS_microseconds
No training or model loading is required.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .evaluation import metrics


LABELS = {1: "antioxidant", 2: "antimicrobial", 3: "toxic"}
METRICS = ["auc_macro_ovr", "accuracy", "sensitivity_macro", "specificity_macro",
           "mcc", "precision_macro", "f1_macro", "f1_weighted", "log_loss"]


def score_predictions(y, probabilities):
    """Three-class metrics; specificity and class accuracy use one-vs-rest."""
    from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score

    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y, predicted, labels=[0, 1, 2])
    classes = []
    for c in range(3):
        tp = int(matrix[c, c])
        fn = int(matrix[c].sum() - tp)
        fp = int(matrix[:, c].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        ratio = lambda a, b: a / b if b else 0.0
        classes.append({
            "label": c + 1, "class": LABELS[c + 1], "n": tp + fn,
            "auc_ovr": float(roc_auc_score(y == c, probabilities[:, c])),
            "accuracy_ovr": ratio(tp + tn, len(y)),
            "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
            "precision": ratio(tp, tp + fp), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "mcc_ovr": float(matthews_corrcoef(y == c, predicted == c)),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        })
    overall = metrics(y, probabilities)
    overall["auc_macro_ovr"] = overall.pop("roc_auc_ovr_macro")
    overall["sensitivity_macro"] = overall.pop("recall_macro")
    overall["specificity_macro"] = float(np.mean([r["specificity"] for r in classes]))
    return {key: overall[key] for key in METRICS}, classes


def read_predictions(path, independent=False):
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError(f"No predictions: {path}")
    y = np.asarray([int(r["true_label"]) - 1 for r in records])
    p = np.asarray([[float(r[f"prob_label_{c}"]) for c in (1, 2, 3)] for r in records])
    folds = np.ones(len(records), dtype=int) if independent else np.asarray([int(r["fold"]) for r in records])
    if (not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(axis=1), 1, atol=1e-5)
            or set(y) != {0, 1, 2} or np.any(folds < 1)):
        raise ValueError(f"Invalid three-class predictions: {path}")
    if not independent and len(set(folds)) < 2:
        raise ValueError(f"At least two folds required: {path}")
    for fold in np.unique(folds):
        if set(y[folds == fold]) != {0, 1, 2}:
            raise ValueError(f"Fold {fold} is missing a class: {path}")
    # Saved probabilities are already normalized. Re-normalizing can perturb
    # tied Random Forest scores and change ROC-AUC at floating-point precision.
    return y, p, folds


def add_sheet(workbook, title, rows):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    sheet = workbook.create_sheet(title)
    fields = list(rows[0])
    sheet.append(fields)
    for row in rows:
        sheet.append([row.get(key) for key in fields])
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="243746")
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if isinstance(cell.value, float):
                cell.number_format = "0.0000"
    for i, key in enumerate(fields, 1):
        width = 100 if key == "description" else min(34, max(16, len(key) + 2))
        sheet.column_dimensions[get_column_letter(i)].width = width
    sheet.freeze_panes = "C2" if title != "Notes" else "A2"
    sheet.auto_filter.ref = sheet.dimensions


def plot_cnn_losses(run_dir):
    # Figure + Agg work without a display and do not change global pyplot state.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    histories = sorted((run_dir / "cnn").glob("fold_*/history.csv"),
                       key=lambda p: int(p.parent.name.split("_")[-1]))
    final_history = run_dir / "cnn" / "final_model" / "history.csv"
    if final_history.exists():
        histories.append(final_history)
    if not histories:
        return []
    overview = Figure(figsize=(12, 3.8 * ((len(histories) + 1) // 2)))
    FigureCanvasAgg(overview)
    axes = overview.subplots((len(histories) + 1) // 2, 2, squeeze=False).ravel()
    paths = []
    for index, history in enumerate(histories):
        with history.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        selection = [r for r in rows if r["phase"] == "selection"]
        refit = [r for r in rows if r["phase"] == "refit"]
        if not selection:
            raise ValueError(f"No selection history: {history}")
        comparable = all(r.get("train_eval_loss") for r in selection)
        train_key = "train_eval_loss" if comparable else "train_loss"
        epochs = [int(r["epoch"]) for r in selection]
        validation = [float(r["validation_loss"]) for r in selection]
        metric_path = history.parent / "metrics.json"
        if metric_path.exists():
            selected = json.loads(metric_path.read_text())["training"]["selected_epochs"]
        else:
            selected = epochs[0]
            best = float("inf")
            for epoch, value in zip(epochs, validation):
                if value < best - 1e-5:
                    selected, best = epoch, value

        def selection_plot(ax):
            ax.plot(epochs, [float(r[train_key]) for r in selection], label=(
                "Train CE (eval mode)" if comparable else "Train weighted CE (legacy)"))
            ax.plot(epochs, validation, label="Validation CE")
            ax.axvline(selected, color="gray", linestyle="--", label=f"Selected epoch: {selected}")
            ax.set(title=f"CNN {history.parent.name}: internal validation", xlabel="Epoch", ylabel="Loss")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc="upper right")
            if not comparable:
                ax.text(0.02, 0.02, "Legacy train/validation losses use different weighting.",
                        transform=ax.transAxes, fontsize=8)

        selection_plot(axes[index])
        figure = Figure(figsize=(12, 4.6))
        FigureCanvasAgg(figure)
        left, right = figure.subplots(1, 2)
        selection_plot(left)
        right.plot([int(r["epoch"]) for r in refit],
                   [float(r["train_loss"]) for r in refit], label="Train weighted CE")
        right.set(title=("Refit on full training partition" if history.parent.name == "final_model"
                         else "Refit on outer training fold"), xlabel="Epoch", ylabel="Loss")
        right.text(0.02, 0.02, "Refit has no validation curve.", transform=right.transAxes, fontsize=8)
        right.grid(alpha=0.25)
        right.legend(fontsize=8)
        figure.tight_layout()
        path = history.parent / "loss_curve.png"
        figure.savefig(path, dpi=160)
        paths.append(path)
    for axis in axes[len(histories):]:
        axis.set_visible(False)
    overview.tight_layout()
    path = run_dir / "cnn" / "loss_curves.png"
    overview.savefig(path, dpi=160)
    paths.append(path)
    return paths


def generate_reports(run_dir, models=None):
    from openpyxl import Workbook

    run_dir = Path(run_dir)
    models = models if models is not None else [
        name for name in ("cnn", "rf", "lightgbm")
        if (run_dir / name / "predictions.csv").exists()]
    if not models:
        raise ValueError(f"No model predictions found in {run_dir}")
    summary, folds_table, class_table, class_cv = [], [], [], []
    class_keys = ["auc_ovr", "accuracy_ovr", "sensitivity", "specificity", "precision", "f1", "mcc_ovr"]
    for model in models:
        y, probabilities, folds = read_predictions(run_dir / model / "predictions.csv")
        per_fold, per_class = [], []
        for fold in sorted(set(folds)):
            mask = folds == fold
            scores, classes = score_predictions(y[mask], probabilities[mask])
            per_fold.append(scores)
            per_class.append(classes)
            folds_table.append({"model": model, "fold": int(fold), "n": int(mask.sum()), **scores})
            class_table.extend({"model": model, "evaluation": f"fold_{fold}", **r} for r in classes)
        for aggregation in ("mean", "std"):
            summary.append({"model": model, "evaluation": f"CV_{aggregation}",
                            "n": None, "folds": len(per_fold), **{
                                key: float(np.mean([r[key] for r in per_fold])) if aggregation == "mean"
                                else float(np.std([r[key] for r in per_fold], ddof=1))
                                for key in METRICS}})
            for c in range(3):
                class_cv.append({"model": model, "evaluation": f"CV_{aggregation}",
                                 "label": c + 1, "class": LABELS[c + 1], **{
                                     key: float(np.mean([r[c][key] for r in per_class]))
                                     if aggregation == "mean" else
                                     float(np.std([r[c][key] for r in per_class], ddof=1))
                                     for key in class_keys}})
        scores, classes = score_predictions(y, probabilities)
        summary.append({"model": model, "evaluation": "OOF_pooled", "n": len(y),
                        "folds": len(per_fold), **scores})
        class_table.extend({"model": model, "evaluation": "OOF_pooled", **r} for r in classes)
        test_path = run_dir / model / "test_predictions.csv"
        if test_path.exists():
            test_y, test_probs, _ = read_predictions(test_path, independent=True)
            test_scores, test_classes = score_predictions(test_y, test_probs)
            summary.append({"model": model, "evaluation": "Independent_test", "n": len(test_y),
                            "folds": None, **test_scores})
            class_table.extend({"model": model, "evaluation": "Independent_test", **r}
                               for r in test_classes)
    workbook = Workbook()
    workbook.remove(workbook.active)
    add_sheet(workbook, "Summary", summary)
    add_sheet(workbook, "CV_folds", folds_table)
    add_sheet(workbook, "Class_CV", class_cv)
    add_sheet(workbook, "Class_metrics", class_table)
    notes = {
        "evaluation": "CV_mean/CV_std: 各評価foldの単純平均・標本標準偏差。OOF_pooled: 全評価foldの未学習データ予測を結合した指標。独立testではなく、CVと同じ予測に基づきます。",
        "independent_test": "Independent_testは最初に確保した約10%の独立testを残り約90%で学習した最終モデルで評価。CV・CNNのepoch選択には使いません。この行がない既存実験は独立test未評価です。",
        "holdout": "使用可能な配列を確定後、ラベル比率を保ち乱数seedで配列単位に約90%/10%分割。同一配列の行をまとめるため行数比は厳密な10%とは限りません。holdout_assignments.csvに保存。",
        "labels": "1=抗酸化、2=抗菌、3=毒性。出典の活性ラベルであり、他の活性がないことを保証しません。",
        "auc": "確率を用いたone-vs-rest ROC-AUC。Summaryは3クラスの単純平均。",
        "sensitivity": "TP/(TP+FN)。Summaryはクラス別Sensitivityの単純平均（Macro Recall）。",
        "specificity": "TN/(TN+FP)。各クラスを陽性、残りを陰性としたone-vs-rest。Summaryは3クラスの単純平均。",
        "accuracy": "Summaryは3クラス全体の正解率。Class_metricsのaccuracy_ovrは各クラス対残りの二値正解率。",
        "mcc": "MCはMCC（Matthews相関係数）として実装。Summaryは多クラスMCC、Class_metricsは二値MCC。",
        "zero_division": "Sensitivity・Specificity・Precision・F1の分母が0の場合は0。AUCは各foldに全3クラスがあることを要求。",
        "loss_plot": "selectionは内部学習/検証、refitは外側学習全体での再学習。破線は採用epoch。評価foldのlossによる学習回数選択は行いません。",
        "train_eval_loss": "新しい履歴はepoch終了時・evalモード・重みなしCEを記録し検証lossと比較。既存履歴は重み付き学習lossを使用し図に注記。refitは重み付き学習lossのみ。",
        "overfitting": "学習loss低下と検証loss上昇の継続を確認してください。曲線だけで過学習の有無を自動判定しません。",
    }
    add_sheet(workbook, "Notes", [{"item": k, "description": v} for k, v in notes.items()])
    path = run_dir / "summary.xlsx"
    workbook.save(path)
    workbook.close()
    plots = plot_cnn_losses(run_dir) if "cnn" in models else []
    print(f"Excel summary: {path}; CNN loss plots: {len(plots)}", flush=True)
    return path, plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    generate_reports(args.run_dir)


if __name__ == "__main__":
    main()
