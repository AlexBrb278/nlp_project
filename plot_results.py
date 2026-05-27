import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT        = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

DATA_PATH = RESULTS_DIR / "combined_results.json"

with open(DATA_PATH) as f:
    data = json.load(f)

CONFIGS     = list(data.keys())
NOISE_TYPES = list(next(iter(data.values()))["noise_results"].keys())

_BASE_PALETTE = [
    "#00BFAF",
    "#6C9BCF",
    "#F4845F",
    "#B5E48C",
    "#E8C547",
    "#C77DFF",
    "#FF6B9D",
    "#FF9F1C",
]

def _build_palette(n: int) -> list[str]:
    colours = list(_BASE_PALETTE)
    if n > len(colours):
        import matplotlib.cm as cm
        tab20 = cm.get_cmap("tab20")
        extra = [
            f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
            for r, g, b, _ in (tab20(i / 20) for i in range(20))
            if f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}" not in colours
        ]
        colours.extend(extra)
    return colours[:n]

CONFIG_COLORS = {cfg: c for cfg, c in zip(CONFIGS, _build_palette(len(CONFIGS)))}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})


def acc_matrix(metric: str) -> np.ndarray:
    arr = np.zeros((len(CONFIGS), len(NOISE_TYPES)))
    for r, cfg in enumerate(CONFIGS):
        for c, nt in enumerate(NOISE_TYPES):
            arr[r, c] = data[cfg]["noise_results"][nt][metric] * 100
    return arr


def save(fig, name: str) -> None:
    path = PLOTS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path.relative_to(ROOT)}")


def plot_heatmap(metric: str, title: str, filename: str, cmap: str = "YlOrRd_r") -> None:
    mat = acc_matrix(metric)
    fig, ax = plt.subplots(figsize=(len(NOISE_TYPES) * 1.4 + 1, len(CONFIGS) * 0.8 + 1.2))

    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=mat.min() - 2, vmax=100)

    ax.set_xticks(range(len(NOISE_TYPES)))
    ax.set_yticks(range(len(CONFIGS)))
    ax.set_xticklabels(NOISE_TYPES, rotation=30, ha="right")
    ax.set_yticklabels(CONFIGS)

    for r in range(len(CONFIGS)):
        for c in range(len(NOISE_TYPES)):
            val = mat[r, c]
            color = "white" if val < (mat.max() + mat.min()) / 2 else "black"
            ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label=f"{metric.replace('_', ' ')} (%)")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    save(fig, filename)


plot_heatmap("accuracy", "Accuracy (%) by Config × Noise Type", "accuracy_heatmap.png", "Blues")


def plot_training_curves() -> None:
    metrics = [("loss", "Training Loss"), ("accuracy", "Val Accuracy")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 4.5))

    for ax, (key, ylabel) in zip(axes, metrics):
        for cfg in CONFIGS:
            hist   = data[cfg]["training_history"]
            epochs = [h["epoch"] for h in hist]
            vals   = [h[key] * (100 if key != "loss" else 1) for h in hist]
            ax.plot(epochs, vals, marker="o", markersize=4,
                    label=cfg, color=CONFIG_COLORS[cfg], linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel + (" (%)" if key != "loss" else ""))
        ax.set_title(ylabel, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.5)
        ax.set_xticks(range(1, len(next(iter(data.values()))["training_history"]) + 1))

    fig.suptitle("Training Curves — All Configurations", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "training_curves.png")


plot_training_curves()


def plot_noise_bar() -> None:
    n_cfg   = len(CONFIGS)
    n_noise = len(NOISE_TYPES)
    x       = np.arange(n_noise)
    width   = 0.8 / n_cfg

    fig, ax = plt.subplots(figsize=(max(12, n_noise * 1.8), 5))

    for i, cfg in enumerate(CONFIGS):
        vals = [data[cfg]["noise_results"][nt]["accuracy"] * 100 for nt in NOISE_TYPES]
        ax.bar(x + i * width - (n_cfg - 1) * width / 2, vals,
               width=width * 0.9, label=cfg, color=CONFIG_COLORS[cfg],
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(NOISE_TYPES)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Test Accuracy by Noise Type and Configuration", fontsize=13, fontweight="bold")
    ax.legend(title="Config", framealpha=0.7)
    ax.axhline(90, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    save(fig, "noise_bar.png")


plot_noise_bar()


def plot_degradation() -> None:
    noisy_types = [nt for nt in NOISE_TYPES if nt != "clean"]
    n_cfg       = len(CONFIGS)
    n_noise     = len(noisy_types)
    x           = np.arange(n_noise)
    width       = 0.8 / n_cfg

    fig, ax = plt.subplots(figsize=(max(10, n_noise * 2), 5))

    for i, cfg in enumerate(CONFIGS):
        clean_acc = data[cfg]["noise_results"]["clean"]["accuracy"]
        pdrs = [
            (clean_acc - data[cfg]["noise_results"][nt]["accuracy"]) / clean_acc * 100
            for nt in noisy_types
        ]
        ax.bar(x + i * width - (n_cfg - 1) * width / 2, pdrs,
               width=width * 0.9, label=cfg, color=CONFIG_COLORS[cfg],
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(noisy_types)
    ax.set_ylabel("PDR — Performance Degradation Rate (%)")
    ax.set_title("Accuracy Drop vs Clean Baseline (per Config)", fontsize=13, fontweight="bold")
    ax.legend(title="Config", framealpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    save(fig, "degradation.png")


plot_degradation()


def plot_radar() -> None:
    categories = NOISE_TYPES
    N          = len(categories)
    angles     = [n / float(N) * 2 * math.pi for n in range(N)]
    angles    += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)

    for cfg in CONFIGS:
        vals = [data[cfg]["noise_results"][nt]["accuracy"] * 100 for nt in categories]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=cfg, color=CONFIG_COLORS[cfg])
        ax.fill(angles, vals, alpha=0.08, color=CONFIG_COLORS[cfg])

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color="grey")
    ax.set_title("Accuracy (%) Across Noise Types — Radar", fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), title="Config", fontsize=9)

    fig.tight_layout()
    save(fig, "radar.png")


plot_radar()


def plot_summary() -> None:
    clean_accs = [data[c]["noise_results"]["clean"]["accuracy"] * 100 for c in CONFIGS]
    best_val   = [data[c]["best_val_accuracy"] * 100 for c in CONFIGS]

    x     = np.arange(len(CONFIGS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(CONFIGS) * 1.4), 5))
    b1 = ax.bar(x - width / 2, best_val,   width,
                color=[CONFIG_COLORS[c] for c in CONFIGS], edgecolor="white", alpha=0.6)
    b2 = ax.bar(x + width / 2, clean_accs, width,
                color=[CONFIG_COLORS[c] for c in CONFIGS], edgecolor="white")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8)

    colour_patches = [mpatches.Patch(color=CONFIG_COLORS[c], label=c) for c in CONFIGS]
    style_patches  = [
        mpatches.Patch(facecolor="grey", alpha=0.6, label="Best val accuracy"),
        mpatches.Patch(facecolor="grey", label="Test accuracy (clean)"),
    ]
    ax.legend(handles=colour_patches + style_patches, fontsize=8,
              ncol=2, framealpha=0.7, title="Config / Metric")

    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Summary: Best Val Accuracy vs Test Accuracy per Config",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "summary.png")


plot_summary()

print(f"\nAll plots written to  {PLOTS_DIR.relative_to(ROOT)}")
