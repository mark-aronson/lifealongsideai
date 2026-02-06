"""
Backpropagation: How Networks Learn
====================================

In file 03 we trained a SINGLE neuron with gradient descent.
In file 04 we built a MULTI-LAYER network with hand-tuned weights.

Now we combine both ideas: backpropagation teaches every weight in a
multi-layer network how to adjust, using the chain rule to propagate
error backward from the output to the input.

The result: a network that learns XOR from scratch, starting from
random weights. No hand-tuning, no human rules -- just data and math.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SIGMOID (repeated for self-containment -- see 02 for full gallery)
# =============================================================================

def sigmoid(z):
    """Sigmoid activation: smooth, differentiable step from 0 to 1."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """Derivative of sigmoid: sig(z) * (1 - sig(z))."""
    s = sigmoid(z)
    return s * (1 - s)


# =============================================================================
# NEURAL NETWORK CLASS
# =============================================================================

class NeuralNetwork:
    """
    A fully-connected neural network with sigmoid activations.

    Implements forward pass, backward pass (backpropagation), and
    gradient descent training.

    Parameters
    ----------
    layer_sizes : list of int
        Number of neurons in each layer, e.g. [2, 4, 1] means
        2 inputs, 4 hidden neurons, 1 output.
    seed : int
        Random seed for weight initialization.
    """

    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        for i in range(self.n_layers - 1):
            # Small random weights (scaled by layer size)
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.5
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        """
        Forward pass: compute output for given inputs.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_inputs)
            Input data

        Returns
        -------
        output : ndarray
            Network output
        """
        self.z_values = []    # pre-activation values (for backprop)
        self.a_values = [X]   # post-activation values (inputs are layer 0)

        current = X
        for w, b in zip(self.weights, self.biases):
            z = current @ w + b
            a = sigmoid(z)
            self.z_values.append(z)
            self.a_values.append(a)
            current = a

        return current

    def backward(self, y):
        """
        Backward pass: compute gradients via backpropagation.

        Uses the chain rule to propagate error from output layer
        back through every hidden layer.

        Parameters
        ----------
        y : ndarray, shape (n_samples, n_outputs)
            True target values

        Returns
        -------
        dw_list : list of ndarray
            Gradient of loss w.r.t. each weight matrix
        db_list : list of ndarray
            Gradient of loss w.r.t. each bias vector
        """
        n = len(y)
        n_layers = self.n_layers

        dw_list = [None] * (n_layers - 1)
        db_list = [None] * (n_layers - 1)

        # Start from the output layer
        # dL/da = 2/n * (prediction - target)   for MSE loss
        output = self.a_values[-1]
        delta = (2.0 / n) * (output - y) * sigmoid_derivative(self.z_values[-1])

        # Work backward through layers
        for layer in range(n_layers - 2, -1, -1):
            # Gradient for weights: a_prev^T @ delta
            a_prev = self.a_values[layer]
            dw_list[layer] = a_prev.T @ delta / n
            db_list[layer] = np.mean(delta, axis=0)

            # Propagate delta to previous layer (if not at input)
            if layer > 0:
                delta = (delta @ self.weights[layer].T) * \
                        sigmoid_derivative(self.z_values[layer - 1])

        return dw_list, db_list

    def train(self, X, y, learning_rate=1.0, n_epochs=5000, verbose=True):
        """
        Train the network using gradient descent with backpropagation.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_inputs)
            Training inputs
        y : ndarray, shape (n_samples, n_outputs)
            Training targets
        learning_rate : float
            Step size for gradient descent
        n_epochs : int
            Number of training iterations
        verbose : bool
            Print progress every 10% of epochs

        Returns
        -------
        history : dict
            Training history with losses and weight snapshots
        """
        history = {
            'losses': [],
            'weight_snapshots': [],
            'boundary_snapshots': [],
        }

        for epoch in range(n_epochs):
            # Forward pass
            output = self.forward(X)

            # Compute loss
            loss = np.mean((output - y) ** 2)
            history['losses'].append(loss)

            # Backward pass
            dw_list, db_list = self.backward(y)

            # Update weights
            for i in range(len(self.weights)):
                self.weights[i] -= learning_rate * dw_list[i]
                self.biases[i] -= learning_rate * db_list[i]

            # Record snapshots at key epochs
            if epoch in [0, n_epochs // 4, n_epochs // 2,
                         3 * n_epochs // 4, n_epochs - 1]:
                history['boundary_snapshots'].append(
                    (epoch, [w.copy() for w in self.weights],
                     [b.copy() for b in self.biases])
                )

            # Store weight snapshots periodically
            if epoch % 100 == 0:
                history['weight_snapshots'].append(
                    (epoch, [w.copy() for w in self.weights])
                )

            # Print progress
            if verbose and (epoch % (n_epochs // 10) == 0 or
                            epoch == n_epochs - 1):
                pred = (output >= 0.5).astype(float)
                accuracy = np.mean(pred == y) * 100
                print(f"    Epoch {epoch:5d}: loss={loss:.6f}, "
                      f"accuracy={accuracy:.0f}%")

        return history

    def predict(self, X):
        """
        Make predictions (forward pass with thresholding).

        Parameters
        ----------
        X : ndarray
            Input data

        Returns
        -------
        ndarray
            Binary predictions (0 or 1)
        """
        output = self.forward(X)
        return (output >= 0.5).astype(float)


# =============================================================================
# BACKPROP WALKTHROUGH
# =============================================================================

def backprop_walkthrough(net, X, y):
    """
    Print every gradient computation for one training example.

    Makes backpropagation transparent by showing each step of the
    chain rule for a single input.
    """
    # Use first example
    x = X[0:1]
    target = y[0:1]

    print(f"  Input:  {x[0]}")
    print(f"  Target: {target[0]}")

    # Forward pass with detailed output
    output = net.forward(x)
    print(f"\n  Forward pass:")
    for i, (z, a) in enumerate(zip(net.z_values, net.a_values[1:])):
        layer_name = "Hidden" if i < len(net.z_values) - 1 else "Output"
        print(f"    {layer_name} layer {i + 1}:")
        print(f"      z (pre-activation):  {z[0]}")
        print(f"      a (post-activation): {a[0]}")

    print(f"\n  Output: {output[0]}")
    print(f"  Loss:   {np.mean((output - target) ** 2):.6f}")

    # Backward pass
    dw_list, db_list = net.backward(target)
    print(f"\n  Backward pass (gradients):")
    for i in range(len(dw_list)):
        layer_name = "Hidden" if i < len(dw_list) - 1 else "Output"
        print(f"    {layer_name} layer {i + 1}:")
        print(f"      dW shape: {dw_list[i].shape}, "
              f"max|dW|={np.max(np.abs(dw_list[i])):.6f}")
        print(f"      db shape: {db_list[i].shape}, "
              f"max|db|={np.max(np.abs(db_list[i])):.6f}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_progress(X, y, history, net):
    """
    Visualize the training process.

    Panel 1: Loss curve
    Panels 2-5: Decision boundary at epochs 0, 1/4, 1/2, final
    """
    snapshots = history['boundary_snapshots']
    n_snaps = len(snapshots)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Backpropagation: Network Learning XOR From Scratch",
                 fontsize=14, fontweight='bold')

    # Panel 1: Loss curve
    ax = axes[0, 0]
    ax.plot(history['losses'], 'b-', linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Loss Curve")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Mark snapshot epochs on loss curve
    for epoch, _, _ in snapshots:
        if epoch < len(history['losses']):
            ax.axvline(x=epoch, color='red', alpha=0.3, linestyle='--')

    # Panels 2-5+: Decision boundaries at snapshot epochs
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 150),
                         np.linspace(-0.5, 1.5, 150))
    grid = np.c_[xx.ravel(), yy.ravel()]

    snap_axes = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]

    for idx, (epoch, snap_w, snap_b) in enumerate(snapshots):
        if idx >= len(snap_axes):
            break
        ax = snap_axes[idx]

        # Temporarily set network weights to snapshot
        old_w = [w.copy() for w in net.weights]
        old_b = [b.copy() for b in net.biases]
        net.weights = snap_w
        net.biases = snap_b

        output = net.forward(grid)
        zz = output.reshape(xx.shape)

        ax.contourf(xx, yy, zz, levels=20, cmap='RdYlGn', alpha=0.7)
        ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=2)

        for i in range(len(X)):
            color = 'green' if y[i, 0] == 1 else 'red'
            marker = 'o' if y[i, 0] == 1 else 'x'
            ax.scatter(X[i, 0], X[i, 1], c=color, s=150, marker=marker,
                       zorder=5, edgecolors='black', linewidths=2)

        loss_at_epoch = history['losses'][min(epoch, len(history['losses']) - 1)]
        ax.set_title(f"Epoch {epoch}\nloss={loss_at_epoch:.4f}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect('equal')

        # Restore weights
        net.weights = old_w
        net.biases = old_b

    plt.tight_layout()
    plt.show()


def plot_weight_evolution(history):
    """Visualize how weights change during training."""
    if not history['weight_snapshots']:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Weight Evolution During Training",
                 fontsize=14, fontweight='bold')

    for layer_idx, ax in enumerate(axes):
        epochs = []
        weight_values = []
        for epoch, snap_w in history['weight_snapshots']:
            if layer_idx < len(snap_w):
                epochs.append(epoch)
                weight_values.append(snap_w[layer_idx].ravel())

        if weight_values:
            weight_values = np.array(weight_values)
            for w_idx in range(weight_values.shape[1]):
                ax.plot(epochs, weight_values[:, w_idx],
                        linewidth=1.5, alpha=0.8,
                        label=f"w{w_idx}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weight Value")
            layer_name = "Hidden" if layer_idx == 0 else "Output"
            ax.set_title(f"{layer_name} Layer Weights")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    """Run the backpropagation demonstration."""
    print("=" * 70)
    print("BACKPROPAGATION: HOW NETWORKS LEARN")
    print("=" * 70)

    # --- Setup ---
    print("\n--- THE CHALLENGE: LEARN XOR FROM SCRATCH ---")
    print("""
In file 04, we solved XOR with hand-tuned weights.
Now the network must learn XOR from RANDOM initial weights.

Architecture: 2 inputs -> 4 hidden neurons -> 1 output
(4 hidden neurons give the network room to find a solution)

Training data:
  (0,0) -> 0    (0,1) -> 1    (1,0) -> 1    (1,1) -> 0
""")

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    # --- Build and inspect network ---
    net = NeuralNetwork([2, 4, 1], seed=42)

    print("--- INITIAL STATE (RANDOM WEIGHTS) ---")
    print("  Before training, predictions are essentially random:")
    initial_output = net.forward(X)
    for i in range(len(X)):
        print(f"    XOR({X[i,0]:.0f}, {X[i,1]:.0f}) = {initial_output[i,0]:.3f}"
              f"  (expected {y[i,0]:.0f})")

    # --- Backprop walkthrough ---
    print("\n--- BACKPROPAGATION WALKTHROUGH ---")
    print("  Tracing gradient computation for input (0,0):\n")
    backprop_walkthrough(net, X, y)

    # --- Training ---
    print("\n--- TRAINING ---")
    print("  Starting backpropagation training...\n")

    # Re-initialize for clean training
    net = NeuralNetwork([2, 4, 1], seed=42)
    history = net.train(X, y, learning_rate=2.0, n_epochs=5000, verbose=True)

    # --- Test final network ---
    print("\n--- FINAL RESULTS ---")
    final_output = net.forward(X)
    predictions = net.predict(X)
    all_correct = True
    for i in range(len(X)):
        pred = int(predictions[i, 0])
        status = "ok" if pred == y[i, 0] else "WRONG"
        if pred != y[i, 0]:
            all_correct = False
        print(f"  XOR({X[i,0]:.0f}, {X[i,1]:.0f}) = {final_output[i,0]:.4f}"
              f"  -> {pred}  (expected {int(y[i,0])}) {status}")

    if all_correct:
        print("\n  SUCCESS! The network learned XOR from scratch!")
        print("  No hand-tuned weights, no human rules -- just data and math.")

    # --- Visualization ---
    print("\n--- VISUALIZATION ---")
    print("Plotting training progress...")
    plot_training_progress(X, y, history, net)

    print("Plotting weight evolution...")
    plot_weight_evolution(history)

    print("\n" + "=" * 70)
    print("KEY TAKEAWAY")
    print("=" * 70)
    print("""
Backpropagation uses the chain rule to teach every weight in the network:
  1. Forward pass: compute the prediction
  2. Compute loss: how wrong was the prediction?
  3. Backward pass: compute each weight's contribution to the error
  4. Update: adjust each weight to reduce the error

The network learned XOR without being told the rules.
It discovered its own internal representation (the hidden layer)
that makes the problem solvable.

This is the same algorithm that powers ChatGPT, image recognition,
protein folding, and every other deep learning system. The only
differences are scale (billions of parameters) and the details of
architecture -- the core learning principle is identical.
""")


if __name__ == "__main__":
    main()
