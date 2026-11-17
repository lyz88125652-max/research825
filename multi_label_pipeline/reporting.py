"""Export multi-label tables and test ROC/PR/confusion/learning plots."""

import csv
import json
import numpy as np

from . import LABEL_NAMES


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def generate_reports(output, models):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    wb = Workbook(); wb.remove(wb.active)
    def sheet(name, records):
        ws = wb.create_sheet(name)
        if not records: return
        fields = list(records[0]); ws.append(fields)
        for record in records: ws.append([record.get(k) for k in fields])
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="253B4B")
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="center")
                if isinstance(cell.value, float): cell.number_format = "0.0000"
        for i, key in enumerate(fields, 1): ws.column_dimensions[get_column_letter(i)].width = max(15, min(40, len(key) + 2))
        ws.freeze_panes = "C2"; ws.auto_filter.ref = ws.dimensions
    def typed(record):
        result = {}
        integer_fields = {"fold", "train_size", "validation_size", "n", "positives", "negatives", "tp", "fn", "fp", "tn"}
        for key, value in record.items():
            if key in {"model", "evaluation", "label"}: result[key] = value
            elif value == "": result[key] = None
            elif key in integer_fields: result[key] = int(value)
            else: result[key] = float(value)
        return result
    overall, cv_folds, label_rows = [], [], []
    predictions = {}
    for name in models:
        folder = output / name
        summary = json.loads((folder / "summary.json").read_text())
        mean_std = summary["cv_fold_mean_std"]
        metric_names = list(summary["pooled_out_of_fold"])
        for kind in ["mean", "std"]:
            overall.append({"model": name, "evaluation": "CV_" + kind,
                            **{k: mean_std[k + "_" + kind] for k in metric_names}})
        overall.append({"model": name, "evaluation": "OOF_pooled", **summary["pooled_out_of_fold"]})
        test = json.loads((folder / "test_metrics.json").read_text())
        overall.append({"model": name, "evaluation": "Independent_test", **{k: test[k] for k in metric_names}})
        cv_folds += [typed({"model": name, **r}) for r in read_csv(folder / "fold_metrics.csv")]
        for phase in ["oof", "test"]:
            label_rows += [typed({"model": name, "evaluation": phase, **r})
                           for r in read_csv(folder / f"{phase}_label_metrics.csv")]
        records = read_csv(folder / "test_predictions.csv")
        y = np.asarray([[int(r[f"true_{label}"]) for label in LABEL_NAMES] for r in records])
        p = np.asarray([[float(r[f"prob_{label}"]) for label in LABEL_NAMES] for r in records])
        predictions[name] = y, p
    sheet("Summary", overall); sheet("CV_folds", cv_folds); sheet("Label_metrics", label_rows)
    notes = [
        {"item": "task", "definition": "Multi-label: three per-label binary outputs; row probabilities need not sum to one."},
        {"item": "label_order", "definition": ", ".join(LABEL_NAMES)},
        {"item": "zero", "definition": "Not recorded / negative or unknown; treated as zero in this baseline."},
        {"item": "subset_accuracy", "definition": "All three predicted flags must match the true vector."},
        {"item": "label_accuracy_mean", "definition": "Fraction of correct binary decisions; 1 - Hamming loss."},
        {"item": "macro", "definition": "Unweighted average over the three labels."},
        {"item": "micro", "definition": "Aggregate all label decisions before scoring."},
        {"item": "mcc_macro", "definition": "Mean of the three binary label MCC values."},
        {"item": "roc_auc", "definition": "Use positive probabilities per label. Undefined ROC-AUC is left blank/null."},
        {"item": "threshold", "definition": "Fixed in run_config.json; never optimized using the independent test."},
        {"item": "CV_std", "definition": "Sample standard deviation across folds, not a confidence interval."},
    ]
    sheet("Notes", notes)
    wb.save(output / "summary.xlsx"); wb.close()

    palette = ["#2673B8", "#D98A00", "#008264", "#A23B72", "#6B5B95",
               "#C44E52", "#55A868", "#8C564B", "#7F7F7F"]
    colors = dict(zip(models, palette))
    for curve_kind in ["roc", "pr"]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for c, ax in enumerate(axes):
            for name in models:
                y, p = predictions[name]
                stats = read_csv(output / name / "test_label_metrics.csv")[c]
                if len(np.unique(y[:, c])) != 2: continue
                if curve_kind == "roc":
                    x, values, _ = roc_curve(y[:, c], p[:, c])
                    value = float(stats["roc_auc"])
                    ax.plot(x, values, color=colors[name], label=f"{name} AUC={value:.3f}")
                else:
                    precision, recall, _ = precision_recall_curve(y[:, c], p[:, c])
                    value = float(stats["average_precision"])
                    ax.step(recall, precision, where="post", color=colors[name], label=f"{name} AP={value:.3f}")
            if curve_kind == "roc":
                ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
                ax.set(xlabel="False positive rate", ylabel="Sensitivity")
            else:
                prevalence = predictions[models[0]][0][:, c].mean()
                ax.axhline(prevalence, ls="--", color="gray", lw=1)
                ax.set(xlabel="Sensitivity / Recall", ylabel="Precision")
            ax.set(title=LABEL_NAMES[c], xlim=(0, 1), ylim=(0, 1.02))
            ax.legend(fontsize=8, framealpha=1)
        fig.savefig(output / f"test_{curve_kind}.png", dpi=180); plt.close(fig)
    for name in models:
        stats = read_csv(output / name / "test_label_metrics.csv")
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
        for ax, row in zip(axes, stats):
            matrix = np.asarray([[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]])
            denominator = matrix.sum(1, keepdims=True)
            normalized = np.divide(matrix, denominator, out=np.zeros_like(matrix, dtype=float), where=denominator != 0)
            ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                            color="white" if normalized[i, j] > .5 else "black")
            ax.set_xticks([0, 1], ["0", "1"]); ax.set_yticks([0, 1], ["0", "1"])
            ax.set(title=row["label"], xlabel="Predicted", ylabel="True")
        fig.savefig(output / name / "test_confusion_matrices.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    for ax, key, title in zip(axes, ["subset_accuracy", "f1_macro"], ["Exact-match accuracy", "Macro F1"]):
        x = np.arange(len(models))
        means, std, test = [], [], []
        for name in models:
            row = json.loads((output / name / "summary.json").read_text())["cv_fold_mean_std"]
            means.append(row[key + "_mean"]); std.append(row[key + "_std"])
            test.append(json.loads((output / name / "test_metrics.json").read_text())[key])
        ax.errorbar(x - .07, means, yerr=std, fmt="o", capsize=4, label="CV mean ± fold SD")
        ax.scatter(x + .07, test, marker="s", label="Independent test")
        ax.set_xticks(x, models, rotation=35, ha="right")
        ax.set(title=title, ylim=(0, 1.05)); ax.legend(fontsize=8)
    fig.savefig(output / "model_comparison.png", dpi=180); plt.close(fig)

    for neural_name in [name for name in models if name in {"cnn", "mlp"}]:
        histories = sorted((output / neural_name).glob("fold_*/history.csv"))
        histories += [output / neural_name / "final_model/history.csv"]
        fig, axes = plt.subplots((len(histories) + 1) // 2, 2, figsize=(11, 3 * ((len(histories) + 1) // 2)),
                                 squeeze=False, constrained_layout=True)
        for ax, path in zip(axes.flat, histories):
            records = [r for r in read_csv(path) if r["phase"] == "selection"]
            epoch = [int(r["epoch"]) for r in records]
            ax.plot(epoch, [float(r["train_eval_loss"]) for r in records], label="Training (eval mode)")
            ax.plot(epoch, [float(r["validation_loss"]) for r in records], label="Validation")
            selected = json.loads((path.parent / "metrics.json").read_text())["training"]["selected_epochs"]
            ax.axvline(selected, ls="--", color="gray")
            ax.set(title=path.parent.name, xlabel="Epoch", ylabel="Unweighted binary log loss")
            ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(histories):]: ax.set_visible(False)
        fig.savefig(output / neural_name / "loss_curves.png", dpi=180); plt.close(fig)
