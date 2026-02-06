"""
Why Nonlinearity Matters: Activation Functions
===============================================

A single neuron computes: output = f(weighted_sum)

The function f is the "activation function." If f is linear (like f(x) = x),
then stacking 100 layers is mathematically identical to a single layer.
Nonlinear activation functions are what give neural networks their power.

This file explores the most common activation functions and demonstrates
why nonlinearity is essential for learning complex patterns.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def step(z):
    """
    Step function: the original perceptron activation.

    Output is exactly 0 or 1. Simple but not differentiable at z=0,
    which makes learning via gradient descent impossible.

    Parameters
    ----------
    z : float or ndarray
        Input value(s)

    Returns
    -------
    float or ndarray
        1 where z >= 0, else 0
    """
    return np.where(z >= 0, 1.0, 0.0)


def step_derivative(z):
    """Derivative of step function (0 everywhere, undefined at z=0)."""
    return np.zeros_like(z, dtype=float)


def sigmoid(z):
    """
    Sigmoid (logistic) function: a smooth, differentiable step.

    Biological analogy: this resembles a neuron's firing rate curve.
    Low input -> almost no firing. High input -> near-maximum firing.
    The transition is smooth, not abrupt like the step function.

    Parameters
    ----------
    z : float or ndarray
        Input value(s)

    Returns
    -------
    float or ndarray
        Values in (0, 1)
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """
    Derivative of sigmoid: sigmoid(z) * (1 - sigmoid(z)).

    This tells us "which direction to adjust" during learning.
    Note: the derivative is largest near z=0 (where the sigmoid
    is most uncertain) and smallest far from 0 (where it's confident).
    """
    s = sigmoid(z)
    return s * (1 - s)


def tanh_fn(z):
    """
    Hyperbolic tangent: like sigmoid but centered at zero.

    Output range is (-1, 1) instead of (0, 1). This zero-centering
    often helps learning because positive and negative signals are
    treated symmetrically.

    Parameters
    ----------
    z : float or ndarray
        Input value(s)

    Returns
    -------
    float or ndarray
        Values in (-1, 1)
    """
    return np.tanh(z)


def tanh_derivative(z):
    """Derivative of tanh: 1 - tanh(z)^2."""
    return 1.0 - np.tanh(z) ** 2


def relu(z):
    """
    Rectified Linear Unit (ReLU): the modern default.

    Simple rule: if input is positive, pass it through. If negative, output 0.
    Despite its simplicity, ReLU works extremely well in practice and is
    computationally efficient.

    Parameters
    ----------
    z : float or ndarray
        Input value(s)

    Returns
    -------
    float or ndarray
        max(0, z)
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of ReLU: 1 where z > 0, else 0."""
    return np.where(z > 0, 1.0, 0.0)


# =============================================================================
# ACTIVATION FUNCTION GALLERY
# =============================================================================

def plot_activation_functions():
    """
    Plot a 6-panel gallery of activation functions and their derivatives.

    Each row shows a function and its derivative side by side. The derivative
    is important because it tells the network "how much to adjust" each weight.
    """
    z = np.linspace(-6, 6, 500)

    functions = [
        ("Step Function", "The original perceptron",
         step, step_derivative),
        ("Sigmoid", "Smooth step, biological firing rate curve",
         sigmoid, sigmoid_derivative),
        ("Tanh", "Zero-centered sigmoid",
         tanh_fn, tanh_derivative),
        ("ReLU", "Modern default: max(0, z)",
         relu, relu_derivative),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    fig.suptitle("Activation Functions and Their Derivatives",
                 fontsize=14, fontweight='bold')

    for row, (name, subtitle, func, deriv) in enumerate(functions):
        # Function
        ax_f = axes[row, 0]
        ax_f.plot(z, func(z), 'b-', linewidth=2)
        ax_f.axhline(y=0, color='gray', linewidth=0.5)
        ax_f.axvline(x=0, color='gray', linewidth=0.5)
        ax_f.set_title(f"{name}\n{subtitle}", fontsize=10)
        ax_f.set_ylabel("f(z)")
        ax_f.grid(True, alpha=0.3)
        if row == len(functions) - 1:
            ax_f.set_xlabel("z (weighted sum)")

        # Derivative
        ax_d = axes[row, 1]
        ax_d.plot(z, deriv(z), 'r-', linewidth=2)
        ax_d.axhline(y=0, color='gray', linewidth=0.5)
        ax_d.axvline(x=0, color='gray', linewidth=0.5)
        ax_d.set_title(f"Derivative of {name}", fontsize=10)
        ax_d.set_ylabel("f'(z)")
        ax_d.grid(True, alpha=0.3)
        if row == len(functions) - 1:
            ax_d.set_xlabel("z (weighted sum)")

    plt.tight_layout()
    plt.show()


# =============================================================================
# THE LINEARITY PROBLEM
# =============================================================================

def plot_linearity_problem():
    """
    Demonstrate why nonlinearity is essential.

    Left panel: composing linear functions always gives a line.
    Right panel: composing nonlinear functions (sigmoids) can create
    rich, complex shapes -- this is the power of deep networks.
    """
    x = np.linspace(-3, 3, 500)

    fig, (ax_lin, ax_nonlin) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Why Nonlinearity Matters: Linear vs. Nonlinear Composition",
                 fontsize=14, fontweight='bold')

    # --- Left: linear composition ---
    # f(x) = 2x + 1 (layer 1)
    # g(x) = 3x - 2 (layer 2)
    # g(f(x)) = 3(2x + 1) - 2 = 6x + 1 (still a line!)
    layer1_linear = 2 * x + 1
    layer2_linear = 3 * layer1_linear - 2  # = 6x + 1
    equivalent = 6 * x + 1

    ax_lin.plot(x, 2 * x + 1, 'b--', linewidth=1.5, alpha=0.5,
                label="Layer 1: f(x) = 2x + 1")
    ax_lin.plot(x, layer2_linear, 'r-', linewidth=2,
                label="Layer 2: g(f(x)) = 6x + 1")
    ax_lin.plot(x, equivalent, 'k:', linewidth=2,
                label="Single layer: h(x) = 6x + 1")
    ax_lin.set_title("Linear Composition\n"
                     "Two layers = one layer (useless depth!)", fontsize=11)
    ax_lin.set_xlabel("x")
    ax_lin.set_ylabel("output")
    ax_lin.legend(fontsize=9)
    ax_lin.grid(True, alpha=0.3)
    ax_lin.text(0, -15, "No matter how many linear layers\n"
                "you stack, the result is always a line.",
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # --- Right: nonlinear composition ---
    # Composing sigmoid units can create complex shapes
    # Three shifted sigmoids combined
    h1 = sigmoid(5 * (x + 1.5))   # sigmoid centered at -1.5
    h2 = sigmoid(5 * (x - 0.0))   # sigmoid centered at 0
    h3 = sigmoid(5 * (x - 1.5))   # sigmoid centered at 1.5

    # Combine them with different weights to create a bump
    combined = 2.0 * h1 - 3.0 * h2 + 2.0 * h3

    ax_nonlin.plot(x, h1, '--', linewidth=1, alpha=0.4, label="Neuron 1")
    ax_nonlin.plot(x, h2, '--', linewidth=1, alpha=0.4, label="Neuron 2")
    ax_nonlin.plot(x, h3, '--', linewidth=1, alpha=0.4, label="Neuron 3")
    ax_nonlin.plot(x, combined, 'purple', linewidth=3,
                   label="Combined output")
    ax_nonlin.set_title("Nonlinear Composition\n"
                        "Combining sigmoids creates complex shapes",
                        fontsize=11)
    ax_nonlin.set_xlabel("x")
    ax_nonlin.set_ylabel("output")
    ax_nonlin.legend(fontsize=9)
    ax_nonlin.grid(True, alpha=0.3)
    ax_nonlin.text(0, -2.5, "Each extra layer can create\n"
                   "richer, more complex patterns!",
                   ha='center', fontsize=9, style='italic',
                   bbox=dict(boxstyle='round', facecolor='lightgreen'))

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    """Run the activation functions demonstration."""
    print("=" * 70)
    print("WHY NONLINEARITY MATTERS: ACTIVATION FUNCTIONS")
    print("=" * 70)

    # --- Activation Function Gallery ---
    print("\n--- ACTIVATION FUNCTION GALLERY ---")
    print("""
Every artificial neuron computes: output = f(w1*x1 + w2*x2 + ... + bias)

The function f is the "activation function." Here are the most common ones:

  Step:     0 or 1, like a light switch.
            The original perceptron used this. Not differentiable.

  Sigmoid:  Smooth curve from 0 to 1, like a dimmer switch.
            Resembles a biological neuron's firing rate curve.
            Differentiable, so we can compute gradients for learning.

  Tanh:     Like sigmoid but ranges from -1 to +1.
            Zero-centered, which helps learning converge faster.

  ReLU:     max(0, x) -- dead simple, but extremely effective.
            The modern default for most neural networks.
""")
    plot_activation_functions()

    # --- Why Derivatives Matter ---
    print("\n--- WHY DERIVATIVES MATTER ---")
    print("""
The derivative of the activation function tells us:
  "If I nudge the input a tiny bit, how much does the output change?"

This is crucial for learning! When the derivative is:
  - LARGE: the neuron is sensitive to changes (fast learning)
  - SMALL: the neuron is insensitive (slow or stuck learning)
  - ZERO:  the neuron learns nothing (the "vanishing gradient" problem)

Notice that sigmoid's derivative peaks at z=0 (where it's uncertain)
and vanishes far from 0 (where it's already confident). This matters!
""")

    # --- The Linearity Problem ---
    print("\n--- THE LINEARITY PROBLEM ---")
    print("""
Here's the key insight: if we use LINEAR activation (f(x) = x),
then stacking layers is USELESS.

Why? Because composing linear functions gives another linear function:
  Layer 1: f(x) = 2x + 1
  Layer 2: g(y) = 3y - 2
  Combined: g(f(x)) = 3(2x + 1) - 2 = 6x + 1  <-- still a line!

A 100-layer linear network computes exactly the same thing as a
1-layer linear network. All that depth is wasted.

But with NONLINEAR activation (like sigmoid), each layer can create
new shapes that the next layer builds upon. This is what makes
deep learning powerful.
""")
    plot_linearity_problem()

    # --- Math proof ---
    print("\n--- PROOF: LINEAR COMPOSITION IS LINEAR ---")
    print("""
  Layer 1: y = W1 * x + b1      (a linear transformation)
  Layer 2: z = W2 * y + b2      (another linear transformation)

  Substituting:
    z = W2 * (W1 * x + b1) + b2
    z = (W2 * W1) * x + (W2 * b1 + b2)
    z = W_combined * x + b_combined    <-- still linear!

  With sigmoid activation:
    Layer 1: y = sigmoid(W1 * x + b1)     (nonlinear!)
    Layer 2: z = sigmoid(W2 * y + b2)     (nonlinear!)

    z = sigmoid(W2 * sigmoid(W1 * x + b1) + b2)  <-- NOT simplifiable!

  Each nonlinear layer adds genuine computational power.
""")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAY")
    print("=" * 70)
    print("""
Without nonlinear activation functions, a 100-layer network equals
a 1-layer network. Nonlinearity is what makes depth meaningful.

Each layer with nonlinear activation can create richer representations,
building complex patterns from simple pieces. This connects to
EMERGENCE: complex behavior arising from simple components.
""")


if __name__ == "__main__":
    main()
