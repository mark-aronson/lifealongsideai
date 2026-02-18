"""
Interactive Multi-Layer Neural Network Simulation
==================================================

A network with 2 inputs, two hidden layers (3 neurons each), and 1 output.
Left panel: network diagram with connections colored by weight value.
Right panel: heatmap of the output neuron across the 2-D input space.
Bottom: sliders for all weights and biases, organized by layer.

Use the sliders to adjust weights and biases and watch the heatmap update.
Connection lines are colored red (positive) / blue (negative) and thickened
by magnitude so you can see the weight structure at a glance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Circle

# ── Globals ──────────────────────────────────────────────────────────────────
GRID_RES = 80          # resolution of the heatmap grid
X_RANGE = (-2, 2)
Y_RANGE = (-2, 2)

ACTIVATIONS = {
    "Sigmoid": lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))),
    "ReLU":    lambda z: np.maximum(0, z),
    "Tanh":    lambda z: np.tanh(z),
    "Linear":  lambda z: z,
}
current_activation = ["Sigmoid"]

# ── Initial weights ─────────────────────────────────────────────────────────
# Layer 1  Input(2) → Hidden1(3)   shape (3, 2)
init_W1 = np.array([[ 0.5, -0.5],
                     [ 0.5,  0.5],
                     [-0.5,  0.5]], dtype=float)
init_b1 = np.zeros(3)

# Layer 2  Hidden1(3) → Hidden2(3)   shape (3, 3)
init_W2 = np.array([[ 0.5,  0.0, -0.5],
                     [ 0.0,  0.5,  0.0],
                     [-0.5,  0.0,  0.5]], dtype=float)
init_b2 = np.zeros(3)

# Layer 3  Hidden2(3) → Output(1)   shape (1, 3)
init_W3 = np.array([[0.5, 0.5, 0.5]], dtype=float)
init_b3 = np.zeros(1)


def forward(x1, x2, W1, b1, W2, b2, W3, b3):
    """Forward pass: 2 → 3 → 3 → 1.  Works element-wise on arrays."""
    act = ACTIVATIONS[current_activation[0]]
    # Hidden layer 1
    h1_0 = act(W1[0, 0] * x1 + W1[0, 1] * x2 + b1[0])
    h1_1 = act(W1[1, 0] * x1 + W1[1, 1] * x2 + b1[1])
    h1_2 = act(W1[2, 0] * x1 + W1[2, 1] * x2 + b1[2])
    # Hidden layer 2
    h2_0 = act(W2[0, 0] * h1_0 + W2[0, 1] * h1_1 + W2[0, 2] * h1_2 + b2[0])
    h2_1 = act(W2[1, 0] * h1_0 + W2[1, 1] * h1_1 + W2[1, 2] * h1_2 + b2[1])
    h2_2 = act(W2[2, 0] * h1_0 + W2[2, 1] * h1_1 + W2[2, 2] * h1_2 + b2[2])
    # Output
    return act(W3[0, 0] * h2_0 + W3[0, 1] * h2_1 + W3[0, 2] * h2_2 + b3[0])


# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13))
fig.suptitle("Interactive Multi-Layer Neural Network",
             fontsize=15, fontweight="bold")

ax_net = fig.add_axes([0.02, 0.50, 0.42, 0.44])   # network diagram
ax_map = fig.add_axes([0.52, 0.50, 0.34, 0.44])   # heatmap

# ── Network diagram ─────────────────────────────────────────────────────────
ax_net.set_xlim(-0.5, 5.5)
ax_net.set_ylim(-0.8, 3.8)
ax_net.set_aspect("equal")
ax_net.axis("off")
ax_net.set_title("Network Architecture", fontsize=13, fontweight="bold")

LAYER_X = [0.5, 2.0, 3.5, 5.0]
positions = {
    "in":  [(LAYER_X[0], y) for y in [2.5, 1.0]],
    "h1":  [(LAYER_X[1], y) for y in [3.0, 1.75, 0.5]],
    "h2":  [(LAYER_X[2], y) for y in [3.0, 1.75, 0.5]],
    "out": [(LAYER_X[3], 1.75)],
}

NODE_R = 0.25
_nkw = dict(ec="black", lw=2, zorder=5)
_fc = {"in": "#7ec8e3", "h1": "#98d4a3", "h2": "#98d4a3", "out": "#ffb347"}
_labels = {
    "in":  ["$x_1$", "$x_2$"],
    "h1":  ["$h_1^{(1)}$", "$h_2^{(1)}$", "$h_3^{(1)}$"],
    "h2":  ["$h_1^{(2)}$", "$h_2^{(2)}$", "$h_3^{(2)}$"],
    "out": ["$y$"],
}

for layer, poslist in positions.items():
    for i, pos in enumerate(poslist):
        ax_net.add_patch(Circle(pos, NODE_R, fc=_fc[layer], **_nkw))
        ax_net.text(*pos, _labels[layer][i],
                    ha="center", va="center", fontsize=10, zorder=6)

for x, txt in zip(LAYER_X, ["Input", "Hidden 1", "Hidden 2", "Output"]):
    ax_net.text(x, -0.5, txt, ha="center", fontsize=10, fontweight="bold")

# Connection lines (stored for dynamic recolouring)
conn_lines = {}


def _add_line(p1, p2, key):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    d = np.hypot(dx, dy)
    s = (p1[0] + NODE_R * dx / d, p1[1] + NODE_R * dy / d)
    e = (p2[0] - NODE_R * dx / d, p2[1] - NODE_R * dy / d)
    ln, = ax_net.plot([s[0], e[0]], [s[1], e[1]],
                       color="gray", lw=1.5, zorder=2, alpha=0.5)
    conn_lines[key] = ln


for i, h in enumerate(positions["h1"]):
    for j, inp in enumerate(positions["in"]):
        _add_line(inp, h, f"W1_{i}_{j}")

for i, h2 in enumerate(positions["h2"]):
    for j, h1 in enumerate(positions["h1"]):
        _add_line(h1, h2, f"W2_{i}_{j}")

for j, h2 in enumerate(positions["h2"]):
    _add_line(h2, positions["out"][0], f"W3_0_{j}")

# ── Activation selector ─────────────────────────────────────────────────────
ax_act = fig.add_axes([0.90, 0.68, 0.08, 0.18])
ax_act.set_frame_on(False)
radio_act = RadioButtons(ax_act, list(ACTIVATIONS.keys()), active=0)
for lbl in radio_act.labels:
    lbl.set_fontsize(9)
fig.text(0.94, 0.87, "Activation", fontsize=10, fontweight="bold", ha="center")

# ── Heatmap ──────────────────────────────────────────────────────────────────
x1v = np.linspace(*X_RANGE, GRID_RES)
x2v = np.linspace(*Y_RANGE, GRID_RES)
X1g, X2g = np.meshgrid(x1v, x2v)
Z = forward(X1g, X2g, init_W1, init_b1, init_W2, init_b2, init_W3, init_b3)

ax_map.set_title("Output Heatmap", fontsize=13, fontweight="bold")
img = ax_map.imshow(Z, extent=[*X_RANGE, *Y_RANGE], origin="lower",
                     aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
ax_map.set_xlabel("$x_1$", fontsize=12)
ax_map.set_ylabel("$x_2$", fontsize=12)
cbar = fig.colorbar(img, ax=ax_map, fraction=0.046, pad=0.04)
cbar.set_label("output $y$", fontsize=11)

# ── Sliders — three columns at the bottom ───────────────────────────────────
SL_COLOR = "#d5e8d4"
BIAS_COLOR = "#f0d5d5"
SL_H = 0.017          # slider height
SL_GAP = 0.003        # gap between sliders
W_RANGE = (-3.0, 3.0)

sliders = {}


def _make_col(x0, w, items, title, title_y):
    """Create a vertical column of sliders with a header."""
    fig.text(x0 + w / 2, title_y, title,
             fontsize=10, fontweight="bold", ha="center")
    y = title_y - 0.025
    for key, label, val, color in items:
        ax = fig.add_axes([x0, y, w, SL_H])
        sl = Slider(ax, label, *W_RANGE, valinit=val, color=color)
        sl.label.set_fontsize(7)
        sl.valtext.set_fontsize(7)
        sliders[key] = sl
        y -= SL_H + SL_GAP


# Column 1 — Input → Hidden 1  (6 weights + 3 biases)
c1 = []
for i in range(3):
    for j in range(2):
        c1.append((f"W1_{i}_{j}",
                    f"$w^1_{{{i+1},{j+1}}}$", init_W1[i, j], SL_COLOR))
for i in range(3):
    c1.append((f"b1_{i}",
                f"$b^1_{{{i+1}}}$", init_b1[i], BIAS_COLOR))

# Column 2 — Hidden 1 → Hidden 2  (9 weights + 3 biases)
c2 = []
for i in range(3):
    for j in range(3):
        c2.append((f"W2_{i}_{j}",
                    f"$w^2_{{{i+1},{j+1}}}$", init_W2[i, j], SL_COLOR))
for i in range(3):
    c2.append((f"b2_{i}",
                f"$b^2_{{{i+1}}}$", init_b2[i], BIAS_COLOR))

# Column 3 — Hidden 2 → Output  (3 weights + 1 bias)
c3 = []
for j in range(3):
    c3.append((f"W3_0_{j}",
                f"$w^3_{{{j+1}}}$", init_W3[0, j], SL_COLOR))
c3.append(("b3_0", "$b^3$", init_b3[0], BIAS_COLOR))

TITLE_Y = 0.45
_make_col(0.06, 0.24, c1, r"Input $\rightarrow$ Hidden 1", TITLE_Y)
_make_col(0.38, 0.24, c2, r"Hidden 1 $\rightarrow$ Hidden 2", TITLE_Y)
_make_col(0.70, 0.24, c3, r"Hidden 2 $\rightarrow$ Output", TITLE_Y)


# ── Update logic ─────────────────────────────────────────────────────────────
def _read_weights():
    """Gather current slider values into weight matrices / bias vectors."""
    W1 = np.array([[sliders[f"W1_{i}_{j}"].val for j in range(2)]
                    for i in range(3)])
    b1 = np.array([sliders[f"b1_{i}"].val for i in range(3)])
    W2 = np.array([[sliders[f"W2_{i}_{j}"].val for j in range(3)]
                    for i in range(3)])
    b2 = np.array([sliders[f"b2_{i}"].val for i in range(3)])
    W3 = np.array([[sliders[f"W3_0_{j}"].val for j in range(3)]])
    b3 = np.array([sliders["b3_0"].val])
    return W1, b1, W2, b2, W3, b3


def _color_connections(W1, W2, W3):
    """Colour and thicken each connection line by its weight value."""
    mx = 3.0
    all_keys_weights = (
        [(f"W1_{i}_{j}", W1[i, j]) for i in range(3) for j in range(2)]
        + [(f"W2_{i}_{j}", W2[i, j]) for i in range(3) for j in range(3)]
        + [(f"W3_0_{j}", W3[0, j]) for j in range(3)]
    )
    for key, w in all_keys_weights:
        ln = conn_lines[key]
        ln.set_color(plt.cm.RdBu_r(0.5 + w / (2 * mx)))
        ln.set_linewidth(1.0 + 2.5 * min(abs(w) / mx, 1.0))
        ln.set_alpha(0.3 + 0.7 * min(abs(w) / mx, 1.0))


def update(_=None):
    W1, b1, W2, b2, W3, b3 = _read_weights()
    Z = forward(X1g, X2g, W1, b1, W2, b2, W3, b3)
    img.set_data(Z)

    act = current_activation[0]
    if act == "Sigmoid":
        img.set_clim(0, 1)
    elif act == "Tanh":
        img.set_clim(-1, 1)
    elif act == "ReLU":
        img.set_clim(0, 5)
    else:
        img.set_clim(-5, 5)

    _color_connections(W1, W2, W3)
    fig.canvas.draw_idle()


for sl in sliders.values():
    sl.on_changed(update)


def _select_activation(label):
    current_activation[0] = label
    update()


radio_act.on_clicked(_select_activation)

# Set initial connection colours
_color_connections(init_W1, init_W2, init_W3)

if __name__ == "__main__":
    plt.show()
