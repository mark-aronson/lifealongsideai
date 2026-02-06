"""
Gradient Descent: How a Single Neuron Learns
=============================================

In the previous files, we hand-picked weights for logic gates.
But the whole point of machine learning is that the machine finds
its own weights from DATA.

This file demonstrates the core learning algorithm: gradient descent.
The neuron starts with random weights, measures how wrong it is,
and adjusts its weights to be less wrong -- over and over until it
learns the pattern. This is the Bitter Lesson in action: let the
machine find the solution rather than hand-engineering it.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SIGMOID (repeated here for self-containment -- see 02 for full gallery)
# =============================================================================

def sigmoid(z):
    """Sigmoid activation: smooth, differentiable step from 0 to 1."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """Derivative of sigmoid: sig(z) * (1 - sig(z))."""
    s = sigmoid(z)
    return s * (1 - s)


# =============================================================================
# LOSS FUNCTION: MEASURING HOW WRONG THE NEURON IS
# =============================================================================

def mse_loss(predictions, targets):
    """
    Mean Squared Error -- the most intuitive loss function.

    Measures the average squared difference between predictions and targets.
    Squaring ensures:
      1. Errors are always positive (no cancellation)
      2. Large errors are penalized more than small ones

    Parameters
    ----------
    predictions : ndarray
        What the neuron predicted
    targets : ndarray
        What the correct answer was

    Returns
    -------
    float
        Average squared error
    """
    return np.mean((predictions - targets) ** 2)


# =============================================================================
# GRADIENT COMPUTATION: THE CHAIN RULE
# =============================================================================

def compute_gradients(X, y, weights, bias):
    """
    Compute gradients of the loss with respect to weights and bias.

    This is the chain rule in action:
      dL/dw = dL/dpred * dpred/dz * dz/dw

    Where:
      L = loss = mean((pred - y)^2)
      pred = sigmoid(z)
      z = X @ w + b

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data
    y : ndarray, shape (n_samples,)
        True labels
    weights : ndarray, shape (n_features,)
        Current weights
    bias : float
        Current bias

    Returns
    -------
    dw : ndarray
        Gradient of loss w.r.t. weights
    db : float
        Gradient of loss w.r.t. bias
    loss : float
        Current loss value
    """
    n = len(y)

    # Forward pass: compute prediction
    z = X @ weights + bias           # step 1: weighted sum
    pred = sigmoid(z)                # step 2: activation

    # Loss
    loss = mse_loss(pred, y)

    # Backward pass (chain rule, step by step):
    # dL/dpred = 2/n * (pred - y)       derivative of MSE
    dL_dpred = (2.0 / n) * (pred - y)

    # dpred/dz = sigmoid'(z)             derivative of activation
    dpred_dz = sigmoid_derivative(z)

    # Combine: dL/dz = dL/dpred * dpred/dz   (the chain rule!)
    dL_dz = dL_dpred * dpred_dz

    # dz/dw = X                          derivative of weighted sum w.r.t. weights
    # dL/dw = X^T @ dL_dz                (average over all samples)
    dw = X.T @ dL_dz / n

    # dz/db = 1                          derivative of weighted sum w.r.t. bias
    # dL/db = mean(dL_dz)
    db = np.mean(dL_dz)

    return dw, db, loss


# =============================================================================
# TRAINING LOOP: GRADIENT DESCENT
# =============================================================================

def train_neuron(X, y, learning_rate=1.0, n_epochs=1000, seed=42):
    """
    Train a single sigmoid neuron using gradient descent.

    Starting from random weights, the neuron repeatedly:
      1. Computes predictions (forward pass)
      2. Measures error (loss)
      3. Computes gradients (backward pass)
      4. Updates weights in the direction that reduces error

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training inputs
    y : ndarray, shape (n_samples,)
        Training targets (0 or 1)
    learning_rate : float
        Step size for weight updates (too high = unstable, too low = slow)
    n_epochs : int
        Number of training iterations
    seed : int
        Random seed for reproducibility

    Returns
    -------
    history : dict
        Training history with keys: 'losses', 'weights', 'biases',
        'predictions_over_time'
    weights : ndarray
        Learned weights
    bias : float
        Learned bias
    """
    np.random.seed(seed)
    n_features = X.shape[1]

    # Initialize weights randomly (small values)
    weights = np.random.randn(n_features) * 0.5
    bias = 0.0

    # Track training history
    history = {
        'losses': [],
        'weights': [weights.copy()],
        'biases': [bias],
        'predictions_over_time': [],
    }

    for epoch in range(n_epochs):
        # Compute gradients
        dw, db, loss = compute_gradients(X, y, weights, bias)

        # Update weights (move in the direction that reduces loss)
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        # Record history
        history['losses'].append(loss)
        history['weights'].append(weights.copy())
        history['biases'].append(bias)

        # Record predictions at key epochs
        if epoch % (n_epochs // 10) == 0 or epoch == n_epochs - 1:
            pred = sigmoid(X @ weights + bias)
            history['predictions_over_time'].append(
                (epoch, pred.copy(), weights.copy(), bias)
            )

    return history, weights, bias


# =============================================================================
# VISUALIZATION: LEARNING PROCESS
# =============================================================================

def plot_learning_process(X, y, history, weights, bias):
    """
    4-panel visualization of the learning process.

    Panel 1: Loss curve (error decreasing over time)
    Panel 2: Evolving decision boundary at different epochs
    Panel 3: Weight trajectory in weight space
    Panel 4: Final predictions vs. true labels
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("How a Single Neuron Learns (Gradient Descent)",
                 fontsize=14, fontweight='bold')

    # Panel 1: Loss curve
    ax = axes[0, 0]
    ax.plot(history['losses'], 'b-', linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Loss Curve: Error Decreasing Over Time")
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Panel 2: Evolving decision boundary
    ax = axes[0, 1]
    xx = np.linspace(-0.5, 1.5, 200)
    colors = plt.cm.viridis(np.linspace(0, 1, len(history['predictions_over_time'])))
    for i, (epoch, pred, w, b) in enumerate(history['predictions_over_time']):
        if X.shape[1] == 2:
            # Decision boundary: w1*x1 + w2*x2 + b = 0 -> x2 = -(w1*x1 + b) / w2
            if abs(w[1]) > 1e-6:
                boundary_y = -(w[0] * xx + b) / w[1]
                ax.plot(xx, boundary_y, color=colors[i], linewidth=1.5,
                        alpha=0.7, label=f"Epoch {epoch}")
    # Plot data points
    for j in range(len(y)):
        color = 'green' if y[j] == 1 else 'red'
        marker = 'o' if y[j] == 1 else 'x'
        ax.scatter(X[j, 0], X[j, 1], c=color, s=150, marker=marker,
                   zorder=5, edgecolors='black', linewidths=1.5)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Decision Boundary Evolution")
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 3: Weight trajectory
    ax = axes[1, 0]
    w_history = np.array(history['weights'])
    if w_history.shape[1] >= 2:
        ax.plot(w_history[:, 0], w_history[:, 1], 'b-', alpha=0.5, linewidth=1)
        ax.scatter(w_history[0, 0], w_history[0, 1], c='red', s=100,
                   zorder=5, label='Start', marker='s')
        ax.scatter(w_history[-1, 0], w_history[-1, 1], c='green', s=100,
                   zorder=5, label='End', marker='*')
        ax.set_xlabel("Weight 1")
        ax.set_ylabel("Weight 2")
    ax.set_title("Weight Trajectory in Weight Space")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Final predictions
    ax = axes[1, 1]
    final_pred = sigmoid(X @ weights + bias)
    x_idx = np.arange(len(y))
    ax.bar(x_idx - 0.15, y, 0.3, color='green', alpha=0.6, label='True')
    ax.bar(x_idx + 0.15, final_pred, 0.3, color='blue', alpha=0.6,
           label='Predicted')
    # Labels for each point
    for j in range(len(y)):
        label = f"({X[j,0]:.0f},{X[j,1]:.0f})" if X.shape[1] == 2 else f"x{j}"
        ax.text(j, -0.12, label, ha='center', fontsize=8)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Output")
    ax.set_title("Final Predictions vs. True Labels")
    ax.legend()
    ax.set_ylim(-0.2, 1.3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


# =============================================================================
# LEARNING RATE COMPARISON
# =============================================================================

def plot_learning_rate_comparison(X, y):
    """
    Compare different learning rates on the same problem.

    Shows three regimes:
      - Too low: learning is painfully slow
      - Just right: smooth convergence
      - Too high: unstable, may diverge
    """
    rates = [0.1, 1.0, 5.0, 50.0]
    labels = ["0.1 (too slow)", "1.0 (good)", "5.0 (aggressive)", "50.0 (too high)"]
    colors = ['blue', 'green', 'orange', 'red']

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Learning Rate Comparison",
                 fontsize=14, fontweight='bold')

    for lr, label, color in zip(rates, labels, colors):
        history, _, _ = train_neuron(X, y, learning_rate=lr, n_epochs=500,
                                     seed=42)
        # Clip extremely high losses for visualization
        losses = np.clip(history['losses'], 0, 2.0)
        ax.plot(losses, color=color, linewidth=2, label=f"lr={label}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Effect of Learning Rate on Training")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.6)

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    """Run the gradient descent learning demonstration."""
    print("=" * 70)
    print("GRADIENT DESCENT: HOW A SINGLE NEURON LEARNS")
    print("=" * 70)

    # --- Setup: OR gate as learning target ---
    print("\n--- LEARNING TARGET: THE OR GATE ---")
    print("""
Instead of hand-picking weights (like we did in file 01), we'll let
the neuron LEARN the OR gate from examples.

Training data:
  (0,0) -> 0    (0,1) -> 1    (1,0) -> 1    (1,1) -> 1

The neuron starts with random weights and must discover the solution.
""")

    # Training data: OR gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 1], dtype=float)

    # --- Loss Function ---
    print("\n--- THE LOSS FUNCTION (MSE) ---")
    print("""
Mean Squared Error measures how wrong the neuron is:

    Loss = (1/n) * sum( (prediction - target)^2 )

A random neuron's predictions:
""")
    np.random.seed(42)
    random_weights = np.random.randn(2) * 0.5
    random_bias = 0.0
    random_pred = sigmoid(X @ random_weights + random_bias)
    random_loss = mse_loss(random_pred, y)

    for i in range(len(y)):
        print(f"  Input ({X[i,0]:.0f}, {X[i,1]:.0f}): "
              f"predicted {random_pred[i]:.3f}, target {y[i]:.0f}, "
              f"error^2 = {(random_pred[i] - y[i])**2:.3f}")
    print(f"\n  Mean Squared Error = {random_loss:.4f}")
    print("  Goal: drive this loss as close to 0 as possible.")

    # --- Gradient Computation ---
    print("\n--- THE GRADIENT (CHAIN RULE) ---")
    print("""
The gradient tells us: "which direction should I move each weight
to reduce the loss?"

The chain rule breaks this into steps:
  dL/dw = dL/dpred * dpred/dz * dz/dw

Where:
  dL/dpred   = how loss changes with prediction    (2 * (pred - y))
  dpred/dz   = how prediction changes with z       (sigmoid'(z))
  dz/dw      = how z changes with weight            (= input x)

Each factor is simple. The chain rule just multiplies them together.
""")

    dw, db, loss = compute_gradients(X, y, random_weights, random_bias)
    print(f"  Current weights: [{random_weights[0]:.3f}, {random_weights[1]:.3f}]")
    print(f"  Current bias:    {random_bias:.3f}")
    print(f"  Current loss:    {loss:.4f}")
    print(f"  Gradient (dw):   [{dw[0]:.4f}, {dw[1]:.4f}]")
    print(f"  Gradient (db):   {db:.4f}")
    print("""
  The gradient points "uphill" (toward higher loss).
  We move OPPOSITE to the gradient to reduce the loss.
""")

    # --- Training ---
    print("\n--- TRAINING THE NEURON ---")
    print("Starting gradient descent with learning_rate=1.0, epochs=2000...")

    history, learned_weights, learned_bias = train_neuron(
        X, y, learning_rate=1.0, n_epochs=2000, seed=42
    )

    print(f"\n  Initial loss: {history['losses'][0]:.4f}")
    print(f"  Final loss:   {history['losses'][-1]:.6f}")
    print(f"\n  Learned weights: [{learned_weights[0]:.3f}, {learned_weights[1]:.3f}]")
    print(f"  Learned bias:    {learned_bias:.3f}")

    # Test learned neuron
    final_pred = sigmoid(X @ learned_weights + learned_bias)
    print("\n  Testing learned neuron:")
    for i in range(len(y)):
        pred_label = 1 if final_pred[i] >= 0.5 else 0
        status = "ok" if pred_label == y[i] else "WRONG"
        print(f"    OR({X[i,0]:.0f}, {X[i,1]:.0f}) = {final_pred[i]:.3f} "
              f"(rounds to {pred_label}) {status}")

    print("""
  The neuron found its own weights from data -- no hand-coding needed!
  This is the fundamental insight of machine learning.
""")

    # --- Visualization ---
    print("\n--- VISUALIZING THE LEARNING PROCESS ---")
    plot_learning_process(X, y, history, learned_weights, learned_bias)

    # --- Learning Rate Exploration ---
    print("\n--- LEARNING RATE EXPLORATION ---")
    print("""
The learning rate controls step size:
  Too LOW  (0.1):  Safe but painfully slow
  GOOD     (1.0):  Steady convergence
  HIGH     (5.0):  Fast but risky
  Too HIGH (50.0): Unstable, may never converge

The learning rate is the most important "knob" in training.
""")
    plot_learning_rate_comparison(X, y)

    print("\n" + "=" * 70)
    print("KEY TAKEAWAY")
    print("=" * 70)
    print("""
The neuron learns by:
  1. Making a prediction (forward pass)
  2. Measuring the error (loss function)
  3. Computing the direction to improve (gradient via chain rule)
  4. Taking a small step in that direction (weight update)

Repeat thousands of times, and the neuron finds the answer.

This is gradient descent -- the engine behind all of modern AI.
The machine finds its own solution from data, not from human rules.
That's the Bitter Lesson: search and learning beat hand-engineering.
""")


if __name__ == "__main__":
    main()
