"""
Run this script to generate the architecture diagram for the Springer paper.
Output: docs/architecture_diagram.png

Requirements: pip install matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import os

os.makedirs("docs", exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis("off")
fig.patch.set_facecolor("#0f172a")
ax.set_facecolor("#0f172a")


def draw_box(ax, x, y, w, h, label, sublabel="", color="#1e40af", text_color="white",
             fontsize=11, subfontsize=8.5, radius=0.25):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0.05,rounding_size={radius}",
                         linewidth=1.5, edgecolor=color,
                         facecolor=color + "33",  # 20% opacity fill
                         zorder=3)
    ax.add_patch(box)
    # border glow
    box2 = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          linewidth=3, edgecolor=color,
                          facecolor="none", alpha=0.3, zorder=2)
    ax.add_patch(box2)

    if sublabel:
        ax.text(x, y + 0.13, label, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold", zorder=5)
        ax.text(x, y - 0.22, sublabel, ha="center", va="center",
        color="#94a3b8",
        fontsize=subfontsize, style="italic", zorder=5)
    else:
        ax.text(x, y, label, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold", zorder=5)


def draw_arrow(ax, x1, y1, x2, y2, color="#64748b"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=18),
                zorder=4)


def draw_small_box(ax, x, y, w, h, label, color):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.03,rounding_size=0.12",
                         linewidth=1.2, edgecolor=color,
                         facecolor=color + "44", zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            color="white", fontsize=7.5, fontweight="bold", zorder=5)


# ── Title ────────────────────────────────────────────────────────────────────
ax.text(8, 8.55, "Proposed Architecture: YOLOv11s + BiFPN + WIoU",
        ha="center", va="center", color="white", fontsize=14, fontweight="bold")

# ── 1. Input ─────────────────────────────────────────────────────────────────
draw_box(ax, 1.4, 4.5, 2.0, 1.0,
         "Input Image", "640 × 640 × 3 (RGB)",
         color="#0ea5e9")

# ── 2. Backbone ──────────────────────────────────────────────────────────────
draw_box(ax, 4.2, 4.5, 2.4, 3.2,
         "YOLOv11s Backbone", "C3k2 Blocks",
         color="#6366f1")

# sub-boxes inside backbone
draw_small_box(ax, 4.2, 5.8, 1.8, 0.45, "Stage 1  —  stride 2", "#6366f1")
draw_small_box(ax, 4.2, 5.2, 1.8, 0.45, "Stage 2  —  stride 4", "#6366f1")
draw_small_box(ax, 4.2, 4.6, 1.8, 0.45, "Stage 3  —  stride 8", "#6366f1")
draw_small_box(ax, 4.2, 4.0, 1.8, 0.45, "Stage 4  —  stride 16", "#6366f1")
draw_small_box(ax, 4.2, 3.4, 1.8, 0.45, "Stage 5  —  stride 32", "#6366f1")

# ── 3. Feature Maps labels ────────────────────────────────────────────────────
ax.text(6.05, 5.6,  "P3  80×80",  ha="left", va="center", color="#a5b4fc", fontsize=8)
ax.text(6.05, 4.5,  "P4  40×40",  ha="left", va="center", color="#a5b4fc", fontsize=8)
ax.text(6.05, 3.4,  "P5  20×20",  ha="left", va="center", color="#a5b4fc", fontsize=8)

# arrows backbone → BiFPN
draw_arrow(ax, 5.42, 5.6,  6.7, 5.6)
draw_arrow(ax, 5.42, 4.5,  6.7, 4.5)
draw_arrow(ax, 5.42, 3.4,  6.7, 3.4)

# ── 4. BiFPN ─────────────────────────────────────────────────────────────────
bifpn_box = FancyBboxPatch((6.7, 2.7), 2.6, 3.6,
                            boxstyle="round,pad=0.05,rounding_size=0.25",
                            linewidth=2, edgecolor="#10b981",
                            facecolor="#10b98122", zorder=3)
ax.add_patch(bifpn_box)
ax.text(8.0, 6.05, "BiFPN Neck", ha="center", va="center",
        color="white", fontsize=11, fontweight="bold", zorder=5)
ax.text(8.0, 5.73, "Bidirectional Feature Pyramid", ha="center", va="center",
        color="#94a3b8", fontsize=8, style="italic", zorder=5)
ax.text(8.0, 5.45, "channels = 256  |  2 iterations", ha="center", va="center",
        color="#94a3b8", fontsize=8, zorder=5)

# BiFPN internal arrows (bidirectional visual)
for cy in [5.0, 4.5, 4.0, 3.5]:
    ax.annotate("", xy=(8.9, cy), xytext=(7.1, cy),
                arrowprops=dict(arrowstyle="<->", color="#10b981", lw=1.2, mutation_scale=12),
                zorder=4)

draw_small_box(ax, 8.0, 5.0, 1.5, 0.38, "Weighted Fusion  ↓", "#10b981")
draw_small_box(ax, 8.0, 4.5, 1.5, 0.38, "Weighted Fusion  ↑", "#10b981")
draw_small_box(ax, 8.0, 4.0, 1.5, 0.38, "Weighted Fusion  ↓", "#10b981")
draw_small_box(ax, 8.0, 3.5, 1.5, 0.38, "Scale Weights  (w₁,w₂,w₃)", "#10b981")

# BiFPN formula
ax.text(8.0, 3.0, r"$O = \frac{\sum w_i \cdot I_i}{\sum w_i + \varepsilon}$",
        ha="center", va="center", color="#6ee7b7", fontsize=9, zorder=5)

# arrows BiFPN → Detection Head
draw_arrow(ax, 9.32, 5.6,  10.5, 5.6)
draw_arrow(ax, 9.32, 4.5,  10.5, 4.5)
draw_arrow(ax, 9.32, 3.4,  10.5, 3.4)

# ── 5. Detection Head ─────────────────────────────────────────────────────────
draw_box(ax, 11.5, 4.5, 2.2, 3.2,
         "Detection Head", "Anchor-Free",
         color="#f59e0b")

draw_small_box(ax, 11.5, 5.6, 1.8, 0.45, "Cls + Reg Branches", "#f59e0b")
draw_small_box(ax, 11.5, 4.9, 1.8, 0.45, "Bounding Box Pred.", "#f59e0b")
draw_small_box(ax, 11.5, 4.2, 1.8, 0.45, "Confidence Score", "#f59e0b")
draw_small_box(ax, 11.5, 3.5, 1.8, 0.45, "WIoU Loss", "#ef4444")

# WIoU note
ax.text(11.5, 3.0, "Dynamic Focusing\nGeometrically Hard Boxes",
        ha="center", va="center", color="#fca5a5", fontsize=7.5, zorder=5)

# ── 6. Output ─────────────────────────────────────────────────────────────────
draw_arrow(ax, 12.62, 4.5, 13.5, 4.5)

# Bio output
draw_box(ax, 14.5, 5.5, 1.7, 0.8,
         "Bio", "Biodegradable",
         color="#22c55e", fontsize=11)

# Non-Bio output
draw_box(ax, 14.5, 3.5, 1.7, 0.8,
         "Non-Bio", "Non-Biodegradable",
         color="#ef4444", fontsize=11)

ax.text(14.5, 4.5, "+", ha="center", va="center", color="white", fontsize=16, fontweight="bold")

draw_arrow(ax, 13.5, 4.5, 13.8, 5.5)
draw_arrow(ax, 13.5, 4.5, 13.8, 3.5)

ax.text(14.5, 2.8, "Class + Confidence\n+ Bounding Box",
        ha="center", va="center", color="#94a3b8", fontsize=7.5)

# ── 1→2 arrow ─────────────────────────────────────────────────────────────────
draw_arrow(ax, 2.42, 4.5, 3.0, 4.5)

# ── Bottom metrics bar ────────────────────────────────────────────────────────
metrics_bg = FancyBboxPatch((0.5, 0.2), 15.0, 0.9,
                             boxstyle="round,pad=0.05,rounding_size=0.15",
                             linewidth=1, edgecolor="#334155",
                             facecolor="#1e293b", zorder=3)
ax.add_patch(metrics_bg)

metrics = [
    ("mAP@0.5",      "94.39%", "#22c55e"),
    ("mAP@0.5:0.95", "85.92%", "#3b82f6"),
    ("Precision",    "91.96%", "#a78bfa"),
    ("Recall",       "91.45%", "#fb923c"),
    ("Inference",    "4.0 ms/img", "#38bdf8"),
    ("Params",       "9.4 M",  "#f472b6"),
    ("Model Size",   "19.2 MB","#facc15"),
]

x_start = 1.1
step = 14.0 / len(metrics)
for i, (lbl, val, col) in enumerate(metrics):
    xc = x_start + i * step
    ax.text(xc, 0.82, lbl, ha="center", va="center", color="#94a3b8", fontsize=7.5)
    ax.text(xc, 0.50, val, ha="center", va="center", color=col,
            fontsize=9.5, fontweight="bold")

ax.text(8.0, 0.08, "YOLOv11s + BiFPN + WIoU  |  100 epochs  |  NVIDIA L40S  |  Adamas University",
        ha="center", va="center", color="#475569", fontsize=7.5)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor="#0e40af44", edgecolor="#1e40af", label="Input"),
    mpatches.Patch(facecolor="#6366f144", edgecolor="#6366f1", label="Backbone"),
    mpatches.Patch(facecolor="#10b98122", edgecolor="#10b981", label="BiFPN Neck"),
    mpatches.Patch(facecolor="#f59e0b44", edgecolor="#f59e0b", label="Detection Head"),
    mpatches.Patch(facecolor="#22c55e44", edgecolor="#22c55e", label="Bio Output"),
    mpatches.Patch(facecolor="#ef444444", edgecolor="#ef4444", label="Non-Bio Output"),
]
leg = ax.legend(handles=legend_items, loc="upper left",
                bbox_to_anchor=(0.01, 0.97),
                fontsize=7.5, framealpha=0.3,
                facecolor="#1e293b", edgecolor="#334155",
                labelcolor="white", ncol=3)

out_path = os.path.join("docs", "architecture_diagram.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"[OK] Architecture diagram saved to: {out_path}")
print("     Use this image in your Springer paper (Figure 1)")
