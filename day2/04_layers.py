"""
Layers and Forward Propagation
===============================

A single neuron can only draw a straight line. To solve XOR (and every
other interesting problem), we need LAYERS of neurons.

The key idea: a hidden layer transforms the data into a NEW representation
where the problem becomes linearly separable. The output layer then draws
a simple line through that transformed space.

In this file we solve XOR with hand-tuned weights to focus on WHAT
layers do, separate from HOW they learn (that's file 05).
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SIGMOID (repeated for self-containment -- see 02 for full gallery)
# =============================================================================

def sigmoid(z):
    """Sigmoid activation: smooth, differentiable step from 0 to 1."""
    return 1.0 / (1.0 + np.exp(-z))


# =============================================================================
# FORWARD PROPAGATION
# =============================================================================

def forward_layer(inputs, weights, biases):
    """
    Compute the output of a single layer of neurons.

    Each neuron in the layer computes: sigmoid(inputs @ weights + bias)

    Parameters
    ----------
    inputs : ndarray, shape (n_samples, n_inputs)
        Input data (or output from previous layer)
    weights : ndarray, shape (n_inputs, n_neurons)
        Weight matrix (one column per neuron)
    biases : ndarray, shape (n_neurons,)
        Bias vector (one per neuron)

    Returns
    -------
    ndarray, shape (n_samples, n_neurons)
        Activated output of this layer
    """
    z = inputs @ weights + biases   # weighted sum for every neuron
    return sigmoid(z)               # apply activation


def forward_network(inputs, network):
    """
    Pass data through an entire network (multiple layers).

    Parameters
    ----------
    inputs : ndarray
        Input data
    network : list of (weights, biases) tuples
        Each element is a layer: (weight_matrix, bias_vector)

    Returns
    -------
    output : ndarray
        Final output of the network
    activations : list of ndarray
        Output of each layer (useful for visualization)
    """
    activations = [inputs]
    current = inputs

    for weights, biases in network:
        current = forward_layer(current, weights, biases)
        activations.append(current)

    return current, activations


# =============================================================================
# XOR SOLVED WITH HAND-TUNED WEIGHTS
# =============================================================================

def build_xor_network():
    """
    Build a 2-layer network that solves XOR using hand-tuned weights.

    Architecture: 2 inputs -> 2 hidden neurons -> 1 output neuron

    Hidden layer strategy:
      - Neuron 1 learns "x1 OR x2" (fires for all except (0,0))
      - Neuron 2 learns "x1 AND x2" (fires only for (1,1))

    Output layer:
      - Fires when neuron 1 is ON but neuron 2 is OFF
      - This is exactly XOR!

    Returns
    -------
    network : list of (weights, biases)
        The hand-tuned XOR network
    """
    # Hidden layer: 2 inputs -> 2 neurons
    # Neuron 1 (OR-like): w=[20, 20], b=-10
    # Neuron 2 (AND-like): w=[20, 20], b=-30
    W_hidden = np.array([[20.0, 20.0],
                         [20.0, 20.0]])
    b_hidden = np.array([-10.0, -30.0])

    # Output layer: 2 hidden -> 1 output
    # Positive weight on OR neuron, negative on AND neuron
    W_output = np.array([[20.0],
                         [-20.0]])
    b_output = np.array([-10.0])

    return [(W_hidden, b_hidden), (W_output, b_output)]


# =============================================================================
# STEP-BY-STEP XOR TRACE
# =============================================================================

def trace_xor_computation(network):
    """
    Print every intermediate value as XOR inputs flow through the network.

    This makes the "black box" transparent: you can see exactly how
    the hidden layer transforms the data.
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    xor_labels = [0, 1, 1, 0]

    W_hidden, b_hidden = network[0]
    W_output, b_output = network[1]

    print("  Step-by-step computation for each XOR input:\n")

    for i in range(len(X)):
        x = X[i]
        expected = xor_labels[i]

        # Hidden layer
        z_hidden = x @ W_hidden + b_hidden
        h = sigmoid(z_hidden)

        # Output layer
        z_output = h @ W_output + b_output
        out = sigmoid(z_output)

        pred = 1 if out[0] >= 0.5 else 0

        print(f"  Input: ({x[0]:.0f}, {x[1]:.0f})  Expected: {expected}")
        print(f"    Hidden z:   [{z_hidden[0]:7.1f}, {z_hidden[1]:7.1f}]")
        print(f"    Hidden out: [{h[0]:.4f}, {h[1]:.4f}]"
              f"  (Neuron1={'ON' if h[0] > 0.5 else 'off'}, "
              f"Neuron2={'ON' if h[1] > 0.5 else 'off'})")
        print(f"    Output z:   {z_output[0]:7.1f}")
        print(f"    Output:     {out[0]:.4f}  ->  {pred}  "
              f"{'ok' if pred == expected else 'WRONG'}")
        print()


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_xor_solution(network):
    """
    4-panel visualization of how layers solve XOR.

    Panel 1: Full decision boundary in input space
    Panel 2: Data in hidden-layer space (the "untangling")
    Panel 3: Hidden neuron 1's individual boundary
    Panel 4: Hidden neuron 2's individual boundary
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle("How Layers Solve XOR",
                 fontsize=14, fontweight='bold')

    # --- Panel 1: Full decision boundary ---
    ax = axes[0, 0]
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                         np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    output, _ = forward_network(grid, network)
    zz = output.reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=20, cmap='RdYlGn', alpha=0.7)
    ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=2)
    for i in range(len(X)):
        color = 'green' if y[i] == 1 else 'red'
        marker = 'o' if y[i] == 1 else 'x'
        ax.scatter(X[i, 0], X[i, 1], c=color, s=200, marker=marker,
                   zorder=5, edgecolors='black', linewidths=2)
    ax.set_title("Network Decision Boundary\n(nonlinear!)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect('equal')

    # --- Panel 2: Hidden space ---
    ax = axes[0, 1]
    _, activations = forward_network(X, network)
    hidden = activations[1]  # hidden layer output

    # Also transform the grid
    _, grid_acts = forward_network(grid, network)
    grid_hidden = grid_acts[1]

    ax.scatter(grid_hidden[:, 0], grid_hidden[:, 1],
               c=output.ravel(), cmap='RdYlGn', alpha=0.1, s=1)
    for i in range(len(X)):
        color = 'green' if y[i] == 1 else 'red'
        marker = 'o' if y[i] == 1 else 'x'
        ax.scatter(hidden[i, 0], hidden[i, 1], c=color, s=200,
                   marker=marker, zorder=5, edgecolors='black', linewidths=2)
        ax.annotate(f"({X[i,0]:.0f},{X[i,1]:.0f})",
                    (hidden[i, 0], hidden[i, 1]),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.set_title("Hidden Layer Space\n(data is now linearly separable!)")
    ax.set_xlabel("Hidden Neuron 1 (OR-like)")
    ax.set_ylabel("Hidden Neuron 2 (AND-like)")

    # --- Panels 3 & 4: Individual hidden neurons ---
    W_hidden, b_hidden = network[0]
    for idx, ax in enumerate([axes[1, 0], axes[1, 1]]):
        w = W_hidden[:, idx]
        b = b_hidden[idx]

        zz_neuron = sigmoid(grid @ w + b).reshape(xx.shape)
        ax.contourf(xx, yy, zz_neuron, levels=20, cmap='Blues', alpha=0.7)
        ax.contour(xx, yy, zz_neuron, levels=[0.5], colors='black',
                   linewidths=2, linestyles='--')
        for i in range(len(X)):
            color = 'green' if y[i] == 1 else 'red'
            marker = 'o' if y[i] == 1 else 'x'
            ax.scatter(X[i, 0], X[i, 1], c=color, s=200, marker=marker,
                       zorder=5, edgecolors='black', linewidths=2)
        role = "OR-like" if idx == 0 else "AND-like"
        ax.set_title(f"Hidden Neuron {idx + 1} ({role})")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def plot_network_capacity():
    """
    Show how increasing hidden layer size increases the complexity
    of decision boundaries the network can learn.
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])

    hidden_sizes = [1, 2, 4, 8]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Network Capacity: More Hidden Neurons = More Complex Boundaries",
                 fontsize=13, fontweight='bold')

    np.random.seed(0)

    for ax, n_hidden in zip(axes, hidden_sizes):
        # Build random network (not trained -- just showing capacity)
        W1 = np.random.randn(2, n_hidden) * 5
        b1 = np.random.randn(n_hidden) * 2
        W2 = np.random.randn(n_hidden, 1) * 5
        b2 = np.random.randn(1)
        net = [(W1, b1), (W2, b2)]

        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                             np.linspace(-0.5, 1.5, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        output, _ = forward_network(grid, net)
        zz = output.reshape(xx.shape)

        ax.contourf(xx, yy, zz, levels=20, cmap='RdYlBu', alpha=0.7)
        ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=1.5)
        for i in range(len(X)):
            color = 'green' if y[i] == 1 else 'red'
            marker = 'o' if y[i] == 1 else 'x'
            ax.scatter(X[i, 0], X[i, 1], c=color, s=100, marker=marker,
                       zorder=5, edgecolors='black', linewidths=1.5)
        ax.set_title(f"{n_hidden} hidden neuron{'s' if n_hidden > 1 else ''}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    """Run the layers and forward propagation demonstration."""
    print("=" * 70)
    print("LAYERS AND FORWARD PROPAGATION")
    print("=" * 70)

    # --- The Problem ---
    print("\n--- THE PROBLEM: XOR REVISITED ---")
    print("""
In file 01, we showed that a single neuron CANNOT solve XOR.
The reason: XOR is not linearly separable -- no single straight line
can separate the 1s from the 0s.

The solution: add a HIDDEN LAYER of neurons between input and output.

Architecture:
    Input (2) -> Hidden Layer (2 neurons) -> Output (1 neuron)

The hidden layer transforms the data into a new space where XOR
BECOMES linearly separable.
""")

    # --- Build the network ---
    network = build_xor_network()

    print("\n--- NETWORK ARCHITECTURE ---")
    print("""
  Layer 1 (Hidden): 2 inputs -> 2 neurons
    Neuron 1 (OR-like):  weights=[20, 20], bias=-10
      Fires for (0,1), (1,0), and (1,1) -- basically OR
    Neuron 2 (AND-like): weights=[20, 20], bias=-30
      Fires only for (1,1) -- basically AND

  Layer 2 (Output): 2 hidden -> 1 output
    weights=[20, -20], bias=-10
    Fires when Neuron1=ON and Neuron2=OFF
    This is: OR AND (NOT AND) = XOR!
""")

    # --- Step-by-step trace ---
    print("\n--- STEP-BY-STEP COMPUTATION ---")
    trace_xor_computation(network)

    # --- Test the network ---
    print("--- TESTING THE NETWORK ---")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    output, activations = forward_network(X, network)

    all_correct = True
    for i in range(len(X)):
        pred = 1 if output[i, 0] >= 0.5 else 0
        status = "ok" if pred == y[i] else "WRONG"
        if pred != y[i]:
            all_correct = False
        print(f"  XOR({X[i,0]:.0f}, {X[i,1]:.0f}) = {output[i,0]:.4f} "
              f"-> {pred}  (expected {y[i]}) {status}")

    if all_correct:
        print("\n  XOR SOLVED! The hidden layer untangled the data.")

    # --- The Untangling ---
    print("\n--- THE KEY INSIGHT: UNTANGLING ---")
    hidden = activations[1]
    print("""
The hidden layer transforms the 4 XOR points into a new space:

  Original Space:       Hidden Space:
  (0,0) -> class 0      ({h00_0:.2f}, {h00_1:.2f}) -> class 0
  (0,1) -> class 1      ({h01_0:.2f}, {h01_1:.2f}) -> class 1
  (1,0) -> class 1      ({h10_0:.2f}, {h10_1:.2f}) -> class 1
  (1,1) -> class 0      ({h11_0:.2f}, {h11_1:.2f}) -> class 0

In the hidden space, the classes ARE linearly separable!
The output neuron just draws a simple line through this new space.

This is EMERGENCE: two simple neurons working together create a
capability that neither one has alone.
""".format(
        h00_0=hidden[0, 0], h00_1=hidden[0, 1],
        h01_0=hidden[1, 0], h01_1=hidden[1, 1],
        h10_0=hidden[2, 0], h10_1=hidden[2, 1],
        h11_0=hidden[3, 0], h11_1=hidden[3, 1],
    ))

    # --- Visualization ---
    print("--- VISUALIZATION ---")
    print("Plotting the XOR solution (4 panels)...")
    plot_xor_solution(network)

    # --- Network Capacity ---
    print("\n--- NETWORK CAPACITY ---")
    print("""
More hidden neurons = more complex boundaries the network can create.
Here are random (untrained) networks with different hidden layer sizes.
Notice how the boundary complexity increases with more neurons.
""")
    plot_network_capacity()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAY")
    print("=" * 70)
    print("""
Hidden layers TRANSFORM data into new representations where hard
problems become easy. The output layer then solves the easy version.

But we hand-tuned these weights. Real neural networks LEARN their
weights automatically through backpropagation -- that's next!
""")


if __name__ == "__main__":
    main()
