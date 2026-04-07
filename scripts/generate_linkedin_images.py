"""
Generate LinkedIn-ready summary cards from data/training_results.json.

This keeps the publish assets aligned with the latest experiment metrics:
  - results_table.png
  - loss_curve.png
  - architecture_card.png

The circuit diagram is left untouched because the model architecture itself
did not change across the comparison runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "data" / "training_results.json"
OUTPUT_DIR = BASE_DIR / "linkedin_images"

BG = "#11121d"
PANEL = "#171a2a"
PANEL_ALT = "#13263d"
GRID = "#1ec8f5"
WHITE = "#f2f3f7"
MUTED = "#b4bac7"
CORAL = "#ff7b72"
GREEN = "#3ddc97"
PURPLE = "#8b5cf6"
GOLD = "#f5b342"


def load_results():
    with RESULTS_PATH.open() as f:
        return json.load(f)


def f(value, digits=4):
    return float(round(float(value), digits))


def delta_text(new, old, digits=4, suffix=""):
    delta = f(new - old, digits)
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.{digits}f}{suffix}"


def metric_row(metric, *values):
    return [metric, *values]


def style_axis(ax, title=None):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#3a3f77")
    ax.tick_params(colors=WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    if title:
        ax.set_title(title, color=WHITE, fontsize=17, fontweight="bold", pad=10)


def add_table(ax, title, columns, rows, note=None):
    ax.axis("off")
    ax.set_facecolor(PANEL)
    ax.text(0.0, 1.05, title, transform=ax.transAxes, color=WHITE,
            fontsize=16, fontweight="bold", va="bottom")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.12 if note else 0.0, 1.0, 0.82 if note else 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(1.4)
        if row == 0:
            cell.set_facecolor(GRID)
            cell.get_text().set_color("#081018")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(PANEL if row % 2 else PANEL_ALT)
            cell.get_text().set_color(WHITE if col != 0 else MUTED)
            if col > 0:
                cell.get_text().set_fontweight("bold")

    if note:
        ax.text(0.0, 0.02, note, transform=ax.transAxes, color=CORAL,
                fontsize=12, style="italic", va="bottom")


def make_results_table(data):
    adam = data["adam_results"]
    qng = data["qng_results"]
    full = data["full_dataset_results"]
    mlp = data["mlp_results"]
    svm = data["svm_results"]

    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.05, 1.15], hspace=0.3)

    fig.text(0.5, 0.975, "QENCS Publish Summary — Final Experiment Tables",
             ha="center", va="top", color=WHITE, fontsize=28, fontweight="bold")
    fig.text(0.5, 0.945,
             "Adam vs QNG, 2k subset vs full dataset, and VQC vs classical baselines",
             ha="center", va="top", color=MUTED, fontsize=15)

    rows_1 = [
        metric_row("Accuracy", f"{adam['accuracy']:.4f}", f"{qng['accuracy']:.4f}", delta_text(qng["accuracy"], adam["accuracy"])),
        metric_row("Precision", f"{adam['precision']:.4f}", f"{qng['precision']:.4f}", delta_text(qng["precision"], adam["precision"])),
        metric_row("Recall", f"{adam['recall']:.4f}", f"{qng['recall']:.4f}", delta_text(qng["recall"], adam["recall"])),
        metric_row("F1 Score", f"{adam['f1_score']:.4f}", f"{qng['f1_score']:.4f}", delta_text(qng["f1_score"], adam["f1_score"])),
        metric_row("Train Time", f"{adam['train_time_s']:.1f}s", f"{qng['train_time_s']:.1f}s", f"{qng['train_time_s'] / max(adam['train_time_s'], 0.1):.1f}x slower"),
    ]
    add_table(
        fig.add_subplot(gs[0]),
        "1. Adam vs QNG on the same 2,000-sample subset",
        ["Metric", "Adam", "QNG", "Delta"],
        rows_1,
        note="QNG matched Adam's high-recall pattern but took dramatically longer.",
    )

    rows_2 = [
        metric_row("Accuracy", f"{adam['accuracy']:.4f}", f"{full['accuracy']:.4f}", delta_text(full["accuracy"], adam["accuracy"])),
        metric_row("Precision", f"{adam['precision']:.4f}", f"{full['precision']:.4f}", delta_text(full["precision"], adam["precision"])),
        metric_row("Recall", f"{adam['recall']:.4f}", f"{full['recall']:.4f}", delta_text(full["recall"], adam["recall"])),
        metric_row("F1 Score", f"{adam['f1_score']:.4f}", f"{full['f1_score']:.4f}", delta_text(full["f1_score"], adam["f1_score"])),
        metric_row("Train Time", f"{adam['train_time_s']:.1f}s", f"{full['train_time_s']:.1f}s", f"{full['train_time_s'] / max(adam['train_time_s'], 0.1):.1f}x longer"),
    ]
    full_note = (
        "Full-dataset training was slower and slightly worse at the same hyperparameters."
        if full["f1_score"] < adam["f1_score"]
        else "More data slightly improved the result, but the VQC still stayed close to chance overall."
    )
    add_table(
        fig.add_subplot(gs[1]),
        "2. 2k subset vs full dataset with Adam",
        ["Metric", "2k Subset", "Full Dataset", "Delta"],
        rows_2,
        note=full_note,
    )

    rows_3 = [
        metric_row("Accuracy", f"{adam['accuracy']:.4f}", f"{svm['accuracy']:.4f}", f"{mlp['accuracy']:.4f}"),
        metric_row("Precision", f"{adam['precision']:.4f}", f"{svm['precision']:.4f}", f"{mlp['precision']:.4f}"),
        metric_row("Recall", f"{adam['recall']:.4f}", f"{svm['recall']:.4f}", f"{mlp['recall']:.4f}"),
        metric_row("F1 Score", f"{adam['f1_score']:.4f}", f"{svm['f1_score']:.4f}", f"{mlp['f1_score']:.4f}"),
        metric_row("Train Time", f"{adam['train_time_s']:.1f}s", f"{svm['train_time_s']:.2f}s", f"{mlp['train_time_s']:.1f}s"),
    ]
    add_table(
        fig.add_subplot(gs[2]),
        "3. 2k subset three-way comparison",
        ["Metric", "VQC Adam", "SVM", "MLP"],
        rows_3,
        note="MLP and VQC ended up nearly tied on F1; neither established a meaningful edge over simple classical baselines.",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "results_table.png", dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def make_loss_curve(data):
    adam = data["adam_results"]
    qng = data["qng_results"]
    full = data["full_dataset_results"]

    adam_hist = adam["loss_history"]
    qng_hist = qng["loss_history"]
    full_hist = full["loss_history"]

    epochs_adam = [x["epoch"] for x in adam_hist]
    epochs_qng = [x["epoch"] for x in qng_hist]
    epochs_full = [x["epoch"] for x in full_hist]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor=BG)
    fig.suptitle("QENCS Convergence View — What Changed and What Didn't",
                 color=WHITE, fontsize=26, fontweight="bold", y=0.98)

    ax = axes[0]
    style_axis(ax, "Train Loss on 2k Subset")
    ax.plot(epochs_adam, [x["train_loss"] for x in adam_hist], color=GRID, linewidth=2.8, label="Adam")
    ax.plot(epochs_qng, [x["train_loss"] for x in qng_hist], color=GOLD, linewidth=2.4, label="QNG")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=WHITE)
    ax.grid(True, alpha=0.18, color="#8ea2ff")

    ax = axes[1]
    style_axis(ax, "Test Accuracy on 2k Subset")
    ax.plot(epochs_adam, [x["test_acc"] for x in adam_hist], color=PURPLE, linewidth=2.8, label="Adam")
    ax.plot(epochs_qng, [x["test_acc"] for x in qng_hist], color=CORAL, linewidth=2.4, label="QNG")
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1.5, label="Chance")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.45, max(0.7, max([x["test_acc"] for x in adam_hist + qng_hist]) + 0.03))
    ax.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=WHITE)
    ax.grid(True, alpha=0.18, color="#8ea2ff")

    ax = axes[2]
    style_axis(ax, "Adam Test Accuracy: 2k vs Full Dataset")
    ax.plot(epochs_adam, [x["test_acc"] for x in adam_hist], color=GRID, linewidth=2.8, label="Adam 2k subset")
    ax.plot(epochs_full, [x["test_acc"] for x in full_hist], color=GREEN, linewidth=2.6, label="Adam full dataset")
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1.5, label="Chance")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.45, max(0.7, max([x["test_acc"] for x in adam_hist + full_hist]) + 0.03))
    ax.legend(facecolor=PANEL, edgecolor=MUTED, labelcolor=WHITE)
    ax.grid(True, alpha=0.18, color="#8ea2ff")

    fig.text(
        0.5,
        0.03,
        (
            f"QNG final F1: {qng['f1_score']:.4f} vs Adam {adam['f1_score']:.4f} "
            f"while taking {qng['train_time_s'] / max(adam['train_time_s'], 0.1):.1f}x longer. "
            f"Full-dataset Adam reached F1 {full['f1_score']:.4f}."
        ),
        ha="center",
        color=CORAL,
        fontsize=13,
        style="italic",
    )

    fig.savefig(OUTPUT_DIR / "loss_curve.png", dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def add_box(ax, xy, wh, title, lines, edge, face):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=2.0,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.06, title, transform=ax.transAxes,
            ha="center", va="top", color=edge, fontsize=16, fontweight="bold")
    line_y = y + h - 0.14
    available = max(h - 0.20, 0.08)
    spacing = min(0.055, available / max(len(lines), 1))
    font_size = 11.3 if len(lines) >= 5 else 12
    for line in lines:
        ax.text(x + w / 2, line_y, line, transform=ax.transAxes,
                ha="center", va="top", color=MUTED, fontsize=font_size)
        line_y -= spacing


def make_architecture_card(data):
    adam = data["adam_results"]
    qng = data["qng_results"]
    full = data["full_dataset_results"]
    mlp = data["mlp_results"]
    svm = data["svm_results"]

    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    fig.text(0.5, 0.96, "QENCS — Verified Publish Card",
             ha="center", va="top", color=WHITE, fontsize=24, fontweight="bold")
    fig.text(0.5, 0.925,
             "9-qubit VQC architecture stayed fixed; the publish pass added optimizer, scale, and baseline checks",
             ha="center", va="top", color=MUTED, fontsize=14)

    add_box(ax, (0.03, 0.58), (0.2, 0.28), "INPUT", [
        "9 EEG features",
        "Delta / Theta / Alpha / Beta / Gamma",
        "Plus FocusRatio",
    ], GRID, "#102842")

    add_box(ax, (0.27, 0.58), (0.2, 0.28), "EMBEDDING", [
        "AngleEmbedding",
        "features -> [0, pi]",
        "Shared train-fit scaler",
    ], "#b57ee5", "#21103d")

    add_box(ax, (0.51, 0.58), (0.2, 0.28), "QUANTUM CIRCUIT", [
        "4 StronglyEntanglingLayers",
        "shape (4, 9, 3)",
        "108 trainable quantum params",
        "default.qubit simulator",
    ], GREEN, "#072712")

    add_box(ax, (0.75, 0.58), (0.2, 0.28), "OUTPUT", [
        "9 expval(PauliZ) values",
        "Linear(9,1) readout",
        "0.5 classification threshold",
        "BCEWithLogitsLoss",
    ], GOLD, "#2b1409")

    add_box(ax, (0.03, 0.12), (0.44, 0.34), "FINAL RESULTS", [
        f"Adam 2k: acc {adam['accuracy']:.4f} | F1 {adam['f1_score']:.4f} | {adam['train_time_s']:.1f}s",
        f"QNG 2k: acc {qng['accuracy']:.4f} | F1 {qng['f1_score']:.4f} | {qng['train_time_s']:.1f}s",
        f"Full data Adam: acc {full['accuracy']:.4f} | F1 {full['f1_score']:.4f} | {full['train_time_s']:.1f}s",
        f"SVM: F1 {svm['f1_score']:.4f} | MLP: F1 {mlp['f1_score']:.4f}",
        "Takeaway: no clear quantum advantage demonstrated",
    ], GREEN, "#062312")

    add_box(ax, (0.51, 0.12), (0.44, 0.34), "PUBLISH TAKEAWAYS", [
        f"QNG vs Adam: Delta F1 = {delta_text(qng['f1_score'], adam['f1_score'])}",
        f"QNG cost: {qng['train_time_s'] / max(adam['train_time_s'], 0.1):.1f}x slower",
        f"Full vs 2k: Delta F1 = {delta_text(full['f1_score'], adam['f1_score'])}",
        f"MLP vs VQC: Delta F1 = {delta_text(mlp['f1_score'], adam['f1_score'])}",
        "Conclusion: better features / calibration matter more than optimizer swaps here",
    ], GRID, PANEL)

    fig.savefig(OUTPUT_DIR / "architecture_card.png", dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main():
    data = load_results()
    for key in ("adam_results", "qng_results", "full_dataset_results", "mlp_results", "svm_results"):
        if key not in data:
            raise KeyError(f"Missing '{key}' in {RESULTS_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_results_table(data)
    make_loss_curve(data)
    make_architecture_card(data)
    print(f"Updated LinkedIn cards in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
