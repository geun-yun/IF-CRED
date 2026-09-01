from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

OUT = Path("artifacts/frozen/figures/experimental_design_flowchart")
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(18, 10), facecolor="white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

palette = {
    "data": (0.90, 0.95, 0.99),
    "model": (1.00, 0.95, 0.80),
    "experiment": (0.94, 0.90, 0.98),
    "framework": (0.88, 0.96, 0.87),
    "output": (0.95, 0.95, 0.95),
    "blue": "#4C78A8",
    "gold": "#D9A400",
    "purple": "#8064A2",
    "green": "#4C9A47",
    "grey": "#777777",
    "arrow": "#4A4A4A",
}


def box(x, y, w, h, title, body, fill, edge, *, title_size=15, body_size=13,
        title_color="#1F1F1F", lw=2.0, radius=0.025):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.032, title, ha="left", va="top",
            fontsize=title_size, fontweight="bold", color=title_color, zorder=4)
    ax.text(x + 0.018, y + h - 0.078, body, ha="left", va="top",
            fontsize=body_size, color="#202020", linespacing=1.32, zorder=4)
    return patch


def arrow(start, end, *, connection="arc3,rad=0", lw=2.0, mutation=17):
    a = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, color=palette["arrow"],
        connectionstyle=connection, shrinkA=5, shrinkB=5, zorder=2,
    )
    ax.add_patch(a)
    return a


# Inputs
box(
    0.025, 0.655, 0.215, 0.285,
    "Datasets",
    "D6  Adult Census Income\n"
    "D7  Default of Credit Card Clients\n"
    "D8  Cleveland Heart Disease",
    palette["data"], palette["blue"], body_size=12.8,
)
box(
    0.025, 0.430, 0.215, 0.170,
    "Protected-attribute policy",
    "$A_{-}$  Protected excluded\n"
    "$A_{+}$  Primary protected included",
    palette["data"], palette["blue"], body_size=12.8,
)
box(
    0.025, 0.215, 0.215, 0.160,
    "Repeated evaluation",
    "30 random seeds\n"
    "Stratified 70% train / 30% test",
    palette["data"], palette["blue"], body_size=12.8,
)

# Models and probability outputs
box(
    0.285, 0.545, 0.205, 0.395,
    "Prediction models",
    "M1  Logistic Regression (LR)\n"
    "M2  Multilayer Perceptron (MLP)\n"
    "M3  Gaussian Naïve Bayes (NB)\n"
    "M4  Random Forest (RF)\n"
    "M5  Decision Tree (DT)",
    palette["model"], palette["gold"], body_size=12.5,
)
box(
    0.285, 0.250, 0.205, 0.225,
    "Predictive performance",
    "Accuracy ↑     AUROC ↑\n"
    "Brier score ↓  ECE ↓\n"
    "Calibrated probabilities",
    palette["model"], palette["gold"], body_size=12.5,
)

# Experiments
box(
    0.535, 0.745, 0.255, 0.195,
    "E1  Clean-data baseline",
    "Primary IF-CRED profile\n"
    "Protected-policy comparison\n"
    "Audit-design sensitivity",
    palette["experiment"], palette["purple"], body_size=12.2,
)
box(
    0.535, 0.305, 0.255, 0.385,
    "E2  Synthetic failure injection",
    "S1  Clean dense clusters\n"
    "S2  Contradictory near-duplicates\n"
    "S3  Sparse isolated instances\n"
    "S4  Dominant-neighbour pairs\n"
    "S5  Metric-disagreement geometry\n"
    "S6  Model-family disagreement\n"
    "$\\rho \\in \\{0.05,0.10,\\ldots,0.30\\}$",
    palette["experiment"], palette["purple"], body_size=11.8,
)
box(
    0.535, 0.075, 0.255, 0.175,
    "E3  Framework comparison",
    "IF-CRED responses compared with\n"
    "VF1  Formal verification\n"
    "VF2  Statistical inference\n"
    "VF3  IFT-V",
    palette["experiment"], palette["purple"], body_size=11.7,
)

# Evaluation and reported outputs
box(
    0.835, 0.555, 0.145, 0.385,
    "IF-CRED",
    "C  Coverage\n"
    "D  Distance stability\n"
    "F  Individual fairness\n"
    "M  Model stability\n\n"
    "$V=C\\times D\\times F\\times M$",
    palette["framework"], palette["green"], body_size=11.8,
)
box(
    0.835, 0.235, 0.145, 0.255,
    "Reported results",
    "Component profiles\n"
    "Dose–response curves\n"
    "Mean and 95% CI\n"
    "Paired statistical tests",
    palette["output"], palette["grey"], body_size=11.7,
)
box(
    0.835, 0.075, 0.145, 0.105,
    "Interpretation",
    "Credibility of the\nindividual-fairness conclusion",
    palette["output"], palette["grey"], body_size=11.2,
)

# Flow connectors, drawn behind the boxes.
for sy in (0.797, 0.515, 0.295):
    arrow((0.240, sy), (0.285, 0.742), connection="arc3,rad=0.08", lw=1.8)
arrow((0.490, 0.742), (0.535, 0.842), connection="arc3,rad=-0.08")
arrow((0.490, 0.742), (0.535, 0.505), connection="arc3,rad=0.10")
arrow((0.490, 0.360), (0.835, 0.360), connection="arc3,rad=0")
arrow((0.790, 0.842), (0.835, 0.770), connection="arc3,rad=0.05")
arrow((0.790, 0.505), (0.835, 0.690), connection="arc3,rad=-0.10")
arrow((0.907, 0.555), (0.907, 0.490), connection="arc3,rad=0")
arrow((0.790, 0.162), (0.835, 0.300), connection="arc3,rad=-0.10")
arrow((0.907, 0.235), (0.907, 0.180), connection="arc3,rad=0")

# Small stage labels improve navigation without adding a figure title.
for x, label in ((0.132, "Study inputs"), (0.387, "Model fitting"),
                 (0.662, "Experiments"), (0.907, "Evaluation")):
    ax.text(x, 0.978, label, ha="center", va="top", fontsize=12,
            color="#5A5A5A", fontweight="bold")

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.99)
fig.savefig(OUT / "figure_1_experimental_design_flowchart.png", dpi=300,
            bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "figure_1_experimental_design_flowchart.svg",
            bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "figure_1_experimental_design_flowchart.pdf",
            bbox_inches="tight", facecolor="white")
plt.close(fig)
