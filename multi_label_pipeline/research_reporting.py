"""Figures and tables for development search, nested comparisons and final tests."""

import csv
import json
import numpy as np
from . import LABEL_NAMES


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_search(output, ranking, by_k):
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    for key, label in [("best_macro_f1", "Best"), ("median_macro_f1", "Median"), ("mean_macro_f1", "Mean")]:
        axes[0].plot([row["model_count"] for row in by_k], [row[key] for row in by_k], "o-", label=label)
    axes[0].set(xlabel="Number of models", ylabel="Development OOF Macro F1", ylim=(0, 1))
    axes[0].legend()
    top = ranking[:10]
    axes[1].barh([str(row["rank"]) + ": " + row["candidate"] for row in top][::-1],
                 [row["f1_macro"] for row in top][::-1])
    axes[1].set(xlabel="Development OOF Macro F1", xlim=(0, 1), title="Search ranking; selection-biased")
    axes[1].tick_params(axis="y", labelsize=7)
    fig.savefig(output / "ensemble_search.png", dpi=180); plt.close(fig)


def plot_nested(output):
    plt = plotting()
    with (output / "paired_differences.csv").open(newline="") as handle: rows = list(csv.DictReader(handle))
    roles = sorted({row["role"] for row in rows})
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for index, role in enumerate(roles):
        values = [float(row["delta_macro_f1"]) for row in rows if row["role"] == role]
        ax.scatter([index] * len(values), values, alpha=.6)
        ax.plot(index, np.mean(values), "k_")
    ax.axhline(0, color="gray", ls="--"); ax.set_xticks(range(len(roles)), roles, rotation=35)
    ax.set(ylabel="Macro F1 difference: stacking minus voting", title="Paired outer-fold comparisons; black mark = mean")
    fig.savefig(output / "paired_comparison.png", dpi=180); plt.close(fig)


def plot_length(output, lengths, y):
    plt = plotting()
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bins = np.linspace(0, max(lengths) + 1, 35)
    for column, label in enumerate(LABEL_NAMES):
        ax.hist(lengths[y[:, column] == 1], bins=bins, density=True, histtype="step", label=label)
    ax.legend(); ax.set(xlabel="Peptide length", ylabel="Density", title="Training partition only")
    fig.savefig(output / "length_distributions.png", dpi=180); plt.close(fig)


def final_plots(output, records, prefix="test_"):
    from sklearn.metrics import roc_curve, precision_recall_curve
    plt = plotting()
    for kind in ["roc", "pr"]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for column, ax in enumerate(axes):
            for name, (y, p) in records.items():
                if len(np.unique(y[:, column])) != 2: continue
                if kind == "roc":
                    x, values, _ = roc_curve(y[:, column], p[:, column])
                else:
                    values, x, _ = precision_recall_curve(y[:, column], p[:, column])
                ax.plot(x, values, label=name)
            ax.set(title=LABEL_NAMES[column], xlim=(0, 1), ylim=(0, 1.02),
                   xlabel="False positive rate" if kind == "roc" else "Recall",
                   ylabel="Sensitivity" if kind == "roc" else "Precision")
            if kind == "roc": ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.legend(fontsize=7)
        fig.savefig(output / f"{prefix}{kind}.png", dpi=180); plt.close(fig)
    for name in records:
        with (output / name / f"{prefix}label_metrics.csv").open(newline="") as handle:
            metrics = list(csv.DictReader(handle))
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
        for ax, row in zip(axes, metrics):
            matrix = np.array([[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]])
            ax.imshow(matrix, cmap="Blues")
            for i in range(2):
                for j in range(2): ax.text(j, i, str(matrix[i,j]), ha="center", va="center", color="black")
            ax.set(title=row["label"], xlabel="Predicted", ylabel="True", xticks=[0,1], yticks=[0,1])
        fig.savefig(output / name / f"{prefix}confusion_matrices.png", dpi=180); plt.close(fig)


def neural_plots(output, names):
    plt = plotting()
    for name in names:
        histories = sorted((output/name).glob("fold_*/history.csv"))
        if not histories: continue
        fig, axes = plt.subplots((len(histories)+1)//2,2,figsize=(11,3*((len(histories)+1)//2)),squeeze=False,constrained_layout=True)
        for ax,path in zip(axes.flat,histories):
            with path.open(newline="") as handle: rows=[row for row in csv.DictReader(handle) if row["phase"]=="selection"]
            ax.plot([int(row["epoch"]) for row in rows],[float(row["train_eval_loss"]) for row in rows],label="Training eval")
            ax.plot([int(row["epoch"]) for row in rows],[float(row["validation_loss"]) for row in rows],label="Internal validation")
            selected=json.loads((path.parent/"metrics.json").read_text())["training"]["selected_epochs"]
            ax.axvline(selected,color="gray",ls="--")
            ax.set(title=path.parent.name,xlabel="Epoch",ylabel="Unweighted BCE"); ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(histories):]: ax.set_visible(False)
        fig.savefig(output/name/"loss_curves.png",dpi=180); plt.close(fig)


def workbook(output, tables, notes):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    wb = Workbook(); wb.remove(wb.active)
    for title, rows in {**tables, "Notes": [{"definition": text} for text in notes]}.items():
        ws = wb.create_sheet(title)
        if not rows: continue
        fields = list(rows[0]); ws.append(fields)
        for row in rows: ws.append([row.get(key) for key in fields])
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor="253B4B")
        for column in ws.columns:
            ws.column_dimensions[column[0].column_letter].width = min(48, max(16, len(str(column[0].value)) + 2))
        ws.freeze_panes = "B2"; ws.auto_filter.ref = ws.dimensions
    wb.save(output)
