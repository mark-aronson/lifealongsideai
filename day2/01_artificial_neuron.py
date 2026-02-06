"""
From Biology to Math: The Artificial Neuron
============================================

Yesterday we simulated a biological neuron with ion channels, membrane
potentials, and stochastic gating. Today we simplify all of that into
a mathematical model: the perceptron.

A biological neuron:
  1. Receives signals from dendrites (inputs)
  2. Sums them in the cell body (weighted sum)
  3. Fires if the total exceeds a threshold (activation function)

An artificial neuron does exactly the same thing with arithmetic.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# BIOLOGY-TO-MATH COMPARISON DIAGRAM
# =============================================================================

def plot_biology_vs_math():
    """
    Draw a side-by-side comparison of a biological neuron and its
    mathematical abstraction (the perceptron).
    """
    fig, (ax_bio, ax_math) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("From Biological Neuron to Artificial Neuron",
                 fontsize=14, fontweight='bold')

    # --- Biological neuron (left panel) ---
    ax_bio.set_xlim(-1, 6)
    ax_bio.set_ylim(-2, 4)
    ax_bio.set_aspect('equal')
    ax_bio.axis('off')
    ax_bio.set_title("Biological Neuron", fontsize=12, fontweight='bold')

    # Cell body
    cell_body = plt.Circle((3, 1), 0.8, color='#66b2ff', ec='black', lw=2)
    ax_bio.add_patch(cell_body)
    ax_bio.text(3, 1, "Cell\nBody", ha='center', va='center', fontsize=9,
                fontweight='bold')

    # Dendrites (inputs)
    dendrite_labels = ["Dendrite 1", "Dendrite 2", "Dendrite 3"]
    dendrite_y = [3, 2, 0]
    for label, y in zip(dendrite_labels, dendrite_y):
        ax_bio.annotate("", xy=(2.3, 1 + 0.5 * (y - 1.5) / 1.5),
                        xytext=(0.5, y),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax_bio.text(0.3, y, label, ha='right', va='center', fontsize=8,
                    color='green')

    # Axon (output)
    ax_bio.annotate("", xy=(5.5, 1), xytext=(3.8, 1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax_bio.text(5.6, 1, "Axon\n(output)", ha='left', va='center', fontsize=9,
                color='red')

    # Threshold label
    ax_bio.text(3, -1.2, "Fires if total input > threshold",
                ha='center', fontsize=9, style='italic')

    # --- Artificial neuron (right panel) ---
    ax_math.set_xlim(-1, 7)
    ax_math.set_ylim(-2, 4)
    ax_math.set_aspect('equal')
    ax_math.axis('off')
    ax_math.set_title("Artificial Neuron (Perceptron)", fontsize=12,
                      fontweight='bold')

    # Sum node
    sum_circle = plt.Circle((3, 1), 0.8, color='#ffcc66', ec='black', lw=2)
    ax_math.add_patch(sum_circle)
    ax_math.text(3, 1, "Sum\n+ f", ha='center', va='center', fontsize=11,
                 fontweight='bold')

    # Inputs with weights
    input_labels = [("x1", "w1"), ("x2", "w2"), ("x3", "w3")]
    input_y = [3, 1.5, -0.5]
    for (x_label, w_label), y in zip(input_labels, input_y):
        ax_math.annotate("", xy=(2.2, 1 + 0.4 * (y - 1)),
                         xytext=(0.5, y),
                         arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        ax_math.text(0.3, y, x_label, ha='right', va='center', fontsize=11,
                     color='blue', fontweight='bold')
        mid_x = 1.3
        mid_y = (y + 1 + 0.4 * (y - 1)) / 2
        ax_math.text(mid_x, mid_y + 0.25, w_label, ha='center', fontsize=9,
                     color='purple', fontweight='bold')

    # Output
    ax_math.annotate("", xy=(6, 1), xytext=(3.8, 1),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax_math.text(6.1, 1, "output\ny = f(sum(wi*xi) + b)",
                 ha='left', va='center', fontsize=9, color='red')

    # Formula
    ax_math.text(3, -1.2,
                 "output = step(w1*x1 + w2*x2 + w3*x3 + bias)",
                 ha='center', fontsize=9, style='italic',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                           edgecolor='gray'))

    plt.tight_layout()
    plt.show()


# =============================================================================
# THE PERCEPTRON: WEIGHTED SUM + ACTIVATION
# =============================================================================

def weighted_sum(inputs, weights, bias):
    """
    Compute the weighted sum of inputs plus a bias term.

    This is what the cell body does: combine all incoming signals,
    giving more importance to some (higher weight) than others.

    Parameters
    ----------
    inputs : array-like
        Input values (x1, x2, ..., xn)
    weights : array-like
        Connection strengths (w1, w2, ..., wn)
    bias : float
        Bias term (shifts the decision boundary)

    Returns
    -------
    float
        The weighted sum: sum(wi * xi) + bias
    """
    inputs = np.array(inputs, dtype=float)
    weights = np.array(weights, dtype=float)
    return np.dot(inputs, weights) + bias


def step_function(z):
    """
    Step activation function -- the simplest nonlinearity.

    Like a biological neuron's threshold: if the total input is above
    zero, the neuron fires (1); otherwise it stays silent (0).

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


def artificial_neuron(inputs, weights, bias):
    """
    A complete artificial neuron (perceptron).

    Computes: output = step(sum(wi*xi) + bias)

    Parameters
    ----------
    inputs : array-like
        Input values
    weights : array-like
        Connection weights
    bias : float
        Bias term

    Returns
    -------
    float
        0 or 1
    """
    z = weighted_sum(inputs, weights, bias)
    return step_function(z)


# =============================================================================
# LOGIC GATES AS SINGLE NEURONS
# =============================================================================

def and_gate(x1, x2):
    """AND gate: output 1 only when both inputs are 1."""
    return artificial_neuron([x1, x2], weights=[1.0, 1.0], bias=-1.5)


def or_gate(x1, x2):
    """OR gate: output 1 when at least one input is 1."""
    return artificial_neuron([x1, x2], weights=[1.0, 1.0], bias=-0.5)


def not_gate(x1):
    """NOT gate: output 1 when input is 0, and vice versa."""
    return artificial_neuron([x1], weights=[-1.0], bias=0.5)


def xor_attempt(x1, x2):
    """
    Attempt XOR with a single neuron -- this CANNOT work.

    XOR is 1 when inputs differ. No single line can separate
    (0,0)=0 and (1,1)=0 from (0,1)=1 and (1,0)=1.
    We use weights that get as close as possible but still fail.
    """
    # These weights can't solve XOR -- any choice will misclassify at least one point
    return artificial_neuron([x1, x2], weights=[1.0, -1.0], bias=0.0)


# =============================================================================
# DECISION BOUNDARY VISUALIZATION
# =============================================================================

def plot_decision_boundaries():
    """
    Plot decision boundaries for AND, OR, NOT, and XOR.

    Shows that a single neuron draws a straight line through input space.
    AND, OR, and NOT are linearly separable -- XOR is not.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Logic Gates as Single Neurons: Decision Boundaries",
                 fontsize=14, fontweight='bold')

    # Input combinations for 2-input gates
    inputs_2d = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    gates = [
        ("AND Gate\nweights=[1,1], bias=-1.5", [1.0, 1.0], -1.5,
         [0, 0, 0, 1]),
        ("OR Gate\nweights=[1,1], bias=-0.5", [1.0, 1.0], -0.5,
         [0, 1, 1, 1]),
        ("NOT Gate (x1 axis)\nweight=[-1], bias=0.5", [-1.0, 0.0], 0.5,
         [1, 1, 0, 0]),
        ("XOR Gate (IMPOSSIBLE)\nNo single line works!", [1.0, -1.0], 0.0,
         [0, 1, 1, 0]),
    ]

    for ax, (title, weights, bias, expected) in zip(axes.flat, gates):
        # Create a meshgrid for the decision boundary
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                             np.linspace(-0.5, 1.5, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = step_function(grid @ np.array(weights) + bias)
        zz = zz.reshape(xx.shape)

        # Plot the decision regions
        ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5],
                    colors=['#ffcccc', '#ccffcc'], alpha=0.6)
        ax.contour(xx, yy, zz, levels=[0.5], colors='black',
                   linewidths=2, linestyles='--')

        # Plot the input points
        for i, (x, y) in enumerate(inputs_2d):
            color = 'green' if expected[i] == 1 else 'red'
            marker = 'o' if expected[i] == 1 else 'x'
            ax.scatter(x, y, c=color, s=200, marker=marker, zorder=5,
                       edgecolors='black', linewidths=2)
            # Compute actual neuron output
            actual = step_function(np.dot([x, y], weights) + bias)
            match_str = "ok" if actual == expected[i] else "WRONG"
            ax.annotate(f"({x},{y})->{expected[i]} {match_str}",
                        (x, y), textcoords="offset points",
                        xytext=(10, 10), fontsize=8)

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Highlight XOR failure
    axes[1, 1].text(0.5, -0.35, "A single neuron CANNOT solve XOR!",
                    ha='center', fontsize=9, color='red', fontweight='bold')

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    """Run the artificial neuron demonstration."""
    print("=" * 70)
    print("FROM BIOLOGY TO MATH: THE ARTIFICIAL NEURON")
    print("=" * 70)

    # --- Biology vs Math Diagram ---
    print("\n--- BIOLOGY VS. MATH ---")
    print("""
Yesterday we simulated a biological neuron with:
  - Thousands of stochastic ion channels
  - Membrane capacitance and conductances
  - Complex differential equations

Today we simplify all of that into THREE operations:
  1. Multiply each input by a weight    (synapse strength)
  2. Sum everything up, add a bias      (cell body integration)
  3. Apply a threshold function          (fire or don't fire)

That's it. That's the entire artificial neuron.
""")
    plot_biology_vs_math()

    # --- The Perceptron ---
    print("\n--- THE PERCEPTRON ---")
    print("""
The perceptron formula:
    output = step( w1*x1 + w2*x2 + ... + wn*xn + bias )

Let's test it with a simple example:
  inputs  = [0.6, 0.9]
  weights = [0.5, 0.5]
  bias    = -0.5
""")
    inputs = [0.6, 0.9]
    weights = [0.5, 0.5]
    bias = -0.5
    z = weighted_sum(inputs, weights, bias)
    output = step_function(z)
    print(f"  weighted sum = {weights[0]}*{inputs[0]} + "
          f"{weights[1]}*{inputs[1]} + ({bias}) = {z:.2f}")
    print(f"  step({z:.2f}) = {output:.0f}")
    print(f"  The neuron {'fires!' if output == 1 else 'stays silent.'}")

    # --- Logic Gates ---
    print("\n--- LOGIC GATES AS SINGLE NEURONS ---")
    print("""
By choosing the right weights and bias, a single neuron can compute
basic logic operations. This was one of the first results in AI (1943).
""")
    print("  AND gate (both inputs must be 1):")
    for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        result = and_gate(x1, x2)
        print(f"    AND({x1}, {x2}) = {result:.0f}")

    print("\n  OR gate (at least one input must be 1):")
    for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        result = or_gate(x1, x2)
        print(f"    OR({x1}, {x2}) = {result:.0f}")

    print("\n  NOT gate (flip the input):")
    for x1 in [0, 1]:
        result = not_gate(x1)
        print(f"    NOT({x1}) = {result:.0f}")

    # --- The XOR Problem ---
    print("\n--- THE XOR PROBLEM ---")
    print("""
XOR (exclusive or) outputs 1 when inputs DIFFER:
    XOR(0,0) = 0    XOR(0,1) = 1
    XOR(1,0) = 1    XOR(1,1) = 0

Can a single neuron solve this? Let's try:
""")
    print("  Attempted XOR with weights=[1, -1], bias=0:")
    correct = 0
    for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        expected = x1 ^ x2
        result = xor_attempt(x1, x2)
        match_str = "ok" if result == expected else "WRONG"
        if result == expected:
            correct += 1
        print(f"    XOR({x1}, {x2}) = {result:.0f}  "
              f"(expected {expected}) {match_str}")

    print(f"\n  Result: {correct}/4 correct.")
    print("""
  No matter what weights we choose, a single neuron CANNOT solve XOR.
  Why? Because a single neuron can only draw a STRAIGHT LINE through
  the input space. XOR requires a more complex boundary.

  This was a major limitation discovered by Minsky & Papert in 1969.
  The solution? Use MULTIPLE neurons in LAYERS -- which we'll build up to!
""")

    # --- Decision Boundary Visualization ---
    print("\n--- DECISION BOUNDARY VISUALIZATION ---")
    print("Plotting decision boundaries for all four gates...")
    print("  Green circles (o) = class 1, Red crosses (x) = class 0")
    print("  The dashed line is the neuron's decision boundary.")
    plot_decision_boundaries()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAY")
    print("=" * 70)
    print("""
A single artificial neuron can only draw a STRAIGHT LINE through data.
This works for AND, OR, and NOT -- but fails for XOR.

To solve harder problems, we need to combine neurons into LAYERS.
That's where we're headed next!
""")


if __name__ == "__main__":
    main()
