"""
Interactive Neural Network Simulation
======================================

A simple network with two input neurons and one output neuron.
Left panel: network schematic.  Middle panel: heatmap of the output
neuron's value across the 2-D input space.  Right panel: training
data controls and scorecard.

Use the sliders to adjust weights and bias and watch the heatmap update.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
from matplotlib.patches import FancyArrowPatch, Circle

# ── Globals ──────────────────────────────────────────────────────────────────
GRID_RES = 200
X_RANGE = (-2, 2)
Y_RANGE = (-2, 2)

INIT_W1 = 1.0
INIT_W2 = 1.0
INIT_BIAS = 0.0

# Training data: four corner points
TRAIN_POINTS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])

# Logic-gate targets  (inputs mapped: -1→0, 1→1)
GATE_TARGETS = {
    "AND": np.array([0, 0, 0, 1]),
    "OR":  np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}

# Activation functions
ACTIVATIONS = {
    "Linear":  lambda z: z,
    "Sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "ReLU":    lambda z: np.maximum(0, z),
    "Step":    lambda z: np.where(z >= 0, 1.0, 0.0),
}

# Mutable state
training_visible = [False]
current_gate = ["AND"]
bias_enabled = [True]
current_activation = ["Linear"]


def compute_output(x1, x2, w1, w2, bias):
    """Weighted sum followed by the selected activation."""
    z = w1 * x1 + w2 * x2 + bias
    return ACTIVATIONS[current_activation[0]](z)


# ── Build the figure ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 7))
fig.suptitle("Interactive Neural Network", fontsize=15, fontweight="bold")

# Leave room: bottom for sliders, right for training panel
fig.subplots_adjust(bottom=0.28, left=0.04, right=0.64, wspace=0.35)

ax_net = fig.add_subplot(1, 2, 1)   # network diagram
ax_map = fig.add_subplot(1, 2, 2)   # heatmap

# ── Draw the network schematic ───────────────────────────────────────────────
ax_net.set_xlim(-0.5, 3.5)
ax_net.set_ylim(-1, 3)
ax_net.set_aspect("equal")
ax_net.axis("off")
ax_net.set_title("Network", fontsize=13, fontweight="bold")

# Node positions
pos_in1 = (0.5, 2.0)
pos_in2 = (0.5, 0.5)
pos_out = (2.8, 1.25)

node_radius = 0.3
node_kwargs = dict(ec="black", lw=2, zorder=5)

n_in1 = Circle(pos_in1, node_radius, fc="#7ec8e3", **node_kwargs)
n_in2 = Circle(pos_in2, node_radius, fc="#7ec8e3", **node_kwargs)
n_out = Circle(pos_out, node_radius, fc="#ffb347", **node_kwargs)
for p in (n_in1, n_in2, n_out):
    ax_net.add_patch(p)

ax_net.text(*pos_in1, "$x_1$", ha="center", va="center", fontsize=14, zorder=6)
ax_net.text(*pos_in2, "$x_2$", ha="center", va="center", fontsize=14, zorder=6)
ax_net.text(*pos_out, "$y$",   ha="center", va="center", fontsize=14, zorder=6)

# Arrows
arrow_kw = dict(arrowstyle="-|>", color="black", lw=2, mutation_scale=18)
arrow1 = FancyArrowPatch(
    (pos_in1[0] + node_radius, pos_in1[1]),
    (pos_out[0] - node_radius, pos_out[1]),
    connectionstyle="arc3,rad=-0.05", **arrow_kw,
)
arrow2 = FancyArrowPatch(
    (pos_in2[0] + node_radius, pos_in2[1]),
    (pos_out[0] - node_radius, pos_out[1]),
    connectionstyle="arc3,rad=0.05", **arrow_kw,
)
ax_net.add_patch(arrow1)
ax_net.add_patch(arrow2)

# Weight / bias labels
mid1 = ((pos_in1[0] + pos_out[0]) / 2 - 0.1,
        (pos_in1[1] + pos_out[1]) / 2 + 0.2)
mid2 = ((pos_in2[0] + pos_out[0]) / 2 - 0.1,
        (pos_in2[1] + pos_out[1]) / 2 - 0.25)

w1_text = ax_net.text(*mid1, f"$w_1 = {INIT_W1:.2f}$",
                       fontsize=12, color="#9b59b6", fontweight="bold",
                       ha="center", va="center",
                       bbox=dict(fc="white", ec="none", alpha=0.8))
w2_text = ax_net.text(*mid2, f"$w_2 = {INIT_W2:.2f}$",
                       fontsize=12, color="#9b59b6", fontweight="bold",
                       ha="center", va="center",
                       bbox=dict(fc="white", ec="none", alpha=0.8))
bias_text = ax_net.text(pos_out[0], pos_out[1] - 0.55,
                        f"$b = {INIT_BIAS:.2f}$",
                        fontsize=11, color="#e74c3c", fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(fc="white", ec="none", alpha=0.8))

# Formula (updated dynamically)
formula_text = ax_net.text(
    1.65, -0.7, r"$y = w_1 x_1 + w_2 x_2 + b$",
    fontsize=13, ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray"))

# ── Activation function selector (next to output neuron) ────────────────────
ax_act = fig.add_axes([0.27, 0.46, 0.09, 0.20])
ax_act.set_frame_on(False)
radio_act = RadioButtons(ax_act, list(ACTIVATIONS.keys()), active=0)
for lbl in radio_act.labels:
    lbl.set_fontsize(9)
fig.text(0.315, 0.67, "Activation", fontsize=9, fontweight="bold",
         ha="center", va="bottom")

# ── Heatmap ──────────────────────────────────────────────────────────────────
x1_vals = np.linspace(*X_RANGE, GRID_RES)
x2_vals = np.linspace(*Y_RANGE, GRID_RES)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = compute_output(X1, X2, INIT_W1, INIT_W2, INIT_BIAS)

ax_map.set_title("Output Heatmap", fontsize=13, fontweight="bold")
img = ax_map.imshow(
    Z,
    extent=[*X_RANGE, *Y_RANGE],
    origin="lower",
    aspect="auto",
    cmap="RdBu_r",
    vmin=-5, vmax=5,
)
ax_map.set_xlabel("$x_1$", fontsize=12)
ax_map.set_ylabel("$x_2$", fontsize=12)
cbar = fig.colorbar(img, ax=ax_map, fraction=0.046, pad=0.04)
cbar.set_label("output $y$", fontsize=11)

# Training-data scatter (initially hidden)
dot_scatter = ax_map.scatter([], [], s=180, zorder=10, edgecolors="black", linewidths=1.5)

# ── Sliders ──────────────────────────────────────────────────────────────────
slider_color = "#d5e8d4"
disabled_color = "#e0e0e0"

ax_w1   = fig.add_axes([0.08, 0.15, 0.52, 0.03])
ax_w2   = fig.add_axes([0.08, 0.10, 0.52, 0.03])
ax_bias = fig.add_axes([0.08, 0.05, 0.52, 0.03])

sl_w1   = Slider(ax_w1,   "$w_1$",  -3.0, 3.0, valinit=INIT_W1,   color=slider_color)
sl_w2   = Slider(ax_w2,   "$w_2$",  -3.0, 3.0, valinit=INIT_W2,   color=slider_color)
sl_bias = Slider(ax_bias, "bias",   -3.0, 3.0, valinit=INIT_BIAS,  color=slider_color)

# ── Bias enable/disable checkbox (next to output neuron in schematic) ────────
ax_chk_bias = fig.add_axes([0.215, 0.52, 0.02, 0.035])
ax_chk_bias.set_frame_on(False)
chk_bias = CheckButtons(ax_chk_bias, [""], [True])


def _set_bias_slider_active(active):
    if active:
        sl_bias.poly.set_fc(slider_color)
        ax_bias.set_alpha(1.0)
        sl_bias.valtext.set_color("black")
        sl_bias.label.set_color("black")
    else:
        sl_bias.poly.set_fc(disabled_color)
        ax_bias.set_alpha(0.35)
        sl_bias.valtext.set_color("grey")
        sl_bias.label.set_color("grey")


def toggle_bias(_label):
    bias_enabled[0] = not bias_enabled[0]
    _set_bias_slider_active(bias_enabled[0])
    update()


chk_bias.on_clicked(toggle_bias)

# ── Training data panel (right side of figure) ──────────────────────────────
PANEL_X = 0.68

# "Training Data On" checkbox
ax_chk_train = fig.add_axes([PANEL_X, 0.86, 0.04, 0.04])
chk_train = CheckButtons(ax_chk_train, [""], [False])
fig.text(PANEL_X + 0.045, 0.88, "Training Data", fontsize=12,
         fontweight="bold", va="center")

# Logic-gate radio buttons
fig.text(PANEL_X, 0.82, "Logic Gate:", fontsize=11, fontweight="bold")
ax_radio = fig.add_axes([PANEL_X, 0.67, 0.12, 0.15])
ax_radio.set_frame_on(False)
radio_gate = RadioButtons(ax_radio, ["AND", "OR", "XOR"], active=0)
for lbl in radio_gate.labels:
    lbl.set_fontsize(11)

# Grey-out radio initially (training off)
for lbl in radio_gate.labels:
    lbl.set_color("grey")

# Scorecard axes (text-only, no ticks)
ax_score = fig.add_axes([PANEL_X, 0.28, 0.30, 0.38])
ax_score.axis("off")

# Build the initial scorecard text objects
score_title = ax_score.text(0.0, 1.0, "Scorecard", fontsize=12,
                            fontweight="bold", va="top", color="grey",
                            transform=ax_score.transAxes)

header_str = f"{'Point':>10}  {'Target':>6}  {'Output':>7}  {'Cost':>7}"
score_header = ax_score.text(0.0, 0.88, header_str, fontsize=9,
                             fontfamily="monospace", va="top", color="grey",
                             transform=ax_score.transAxes)
score_sep = ax_score.text(0.0, 0.79, "-" * 42, fontsize=9,
                          fontfamily="monospace", va="top", color="grey",
                          transform=ax_score.transAxes)

score_rows = []
y_positions = [0.70, 0.58, 0.46, 0.34]
for i, y_pos in enumerate(y_positions):
    txt = ax_score.text(0.0, y_pos, "", fontsize=9, fontfamily="monospace",
                        va="top", color="grey", transform=ax_score.transAxes)
    score_rows.append(txt)

score_sep2 = ax_score.text(0.0, 0.25, "-" * 42, fontsize=9,
                           fontfamily="monospace", va="top", color="grey",
                           transform=ax_score.transAxes)
score_total = ax_score.text(0.0, 0.14, "", fontsize=10, fontfamily="monospace",
                            fontweight="bold", va="top", color="grey",
                            transform=ax_score.transAxes)


def _update_training_dots(w1, w2, bias):
    """Refresh scatter dots and scorecard."""
    targets = GATE_TARGETS[current_gate[0]]

    if training_visible[0]:
        colors = ["#a70a0a" if t == 1 else "#111f9c" for t in targets]
        dot_scatter.set_offsets(TRAIN_POINTS)
        dot_scatter.set_facecolors(colors)

        text_color = "black"
    else:
        dot_scatter.set_offsets(np.empty((0, 2)))
        text_color = "grey"

    # Update scorecard colours
    score_title.set_color(text_color)
    score_header.set_color(text_color)
    score_sep.set_color(text_color)
    score_sep2.set_color(text_color)

    total_cost = 0.0
    for i, (pt, tgt) in enumerate(zip(TRAIN_POINTS, targets)):
        out = compute_output(pt[0], pt[1], w1, w2, bias)
        cost = (out - tgt) ** 2
        total_cost += cost
        row_str = (f"({pt[0]:+.0f},{pt[1]:+.0f})"
                   f"      {tgt}    "
                   f"{out:+7.3f}  "
                   f"{cost:7.3f}")
        score_rows[i].set_text(row_str)
        score_rows[i].set_color(text_color)

    score_total.set_text(f"{'Total cost:':>28} {total_cost:7.3f}")
    score_total.set_color(text_color)


def toggle_training(_label):
    training_visible[0] = not training_visible[0]
    # Enable / grey-out gate radio and scorecard
    colour = "black" if training_visible[0] else "grey"
    for lbl in radio_gate.labels:
        lbl.set_color(colour)
    update()


def select_gate(label):
    current_gate[0] = label
    update()


chk_train.on_clicked(toggle_training)
radio_gate.on_clicked(select_gate)


def select_activation(label):
    current_activation[0] = label
    update()


radio_act.on_clicked(select_activation)

# Formula strings per activation
_FORMULAS = {
    "Linear":  r"$y = w_1 x_1 + w_2 x_2 + b$",
    "Sigmoid": r"$y = \sigma(w_1 x_1 + w_2 x_2 + b)$",
    "ReLU":    r"$y = \mathrm{ReLU}(w_1 x_1 + w_2 x_2 + b)$",
    "Step":    r"$y = \mathrm{step}(w_1 x_1 + w_2 x_2 + b)$",
}

# ── Main update function ────────────────────────────────────────────────────
def update(_=None):
    w1   = sl_w1.val
    w2   = sl_w2.val
    bias = sl_bias.val if bias_enabled[0] else 0.0

    Z = compute_output(X1, X2, w1, w2, bias)
    img.set_data(Z)

    # Adjust heatmap colour scale for activation
    act = current_activation[0]
    if act in ("Sigmoid", "Step"):
        img.set_clim(0, 1)
    elif act == "ReLU":
        img.set_clim(0, 5)
    else:
        img.set_clim(-5, 5)

    # Schematic labels
    w1_text.set_text(f"$w_1 = {w1:.2f}$")
    w2_text.set_text(f"$w_2 = {w2:.2f}$")
    if bias_enabled[0]:
        bias_text.set_text(f"$b = {bias:.2f}$")
        bias_text.set_color("#e74c3c")
    else:
        bias_text.set_text("$b$ off")
        bias_text.set_color("grey")

    # Formula
    formula_text.set_text(_FORMULAS[current_activation[0]])

    # Training dots + scorecard
    _update_training_dots(w1, w2, bias)

    fig.canvas.draw_idle()


sl_w1.on_changed(update)
sl_w2.on_changed(update)
sl_bias.on_changed(update)

# Initial scorecard fill
_update_training_dots(INIT_W1, INIT_W2, INIT_BIAS)

if __name__ == "__main__":
    plt.show()
