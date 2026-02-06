"""
Putting It All Together: Handwritten Digit Recognition
=======================================================

Everything we've built -- artificial neurons, activation functions,
gradient descent, layers, and backpropagation -- now comes together
on a REAL problem: recognizing handwritten digits (0-9).

Using only addition, multiplication, and the chain rule, we build a
network that achieves ~95% accuracy on handwriting. No magic, no
black boxes -- just the principles from the previous five files
applied at slightly larger scale.

Architecture: 64 inputs -> 32 hidden -> 16 hidden -> 10 outputs
  - 64 inputs: each 8x8 pixel image flattened
  - 32 + 16 hidden: two layers to learn features
  - 10 outputs: one per digit (0-9), using softmax
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# =============================================================================
# ACTIVATION FUNCTIONS (repeated for self-containment)
# =============================================================================

def sigmoid(z):
    """Sigmoid activation: smooth step from 0 to 1."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    """Derivative of sigmoid given the OUTPUT a (not the input z)."""
    return a * (1 - a)


def softmax(z):
    """
    Softmax: converts raw scores into probabilities that sum to 1.

    Used in the output layer for multi-class classification.
    Each output represents P(digit = k | input).

    Parameters
    ----------
    z : ndarray, shape (n_samples, n_classes)
        Raw output scores

    Returns
    -------
    ndarray, shape (n_samples, n_classes)
        Probability distribution over classes
    """
    # Subtract max for numerical stability (prevents overflow)
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prepare_data(test_size=0.2, seed=42):
    """
    Load the sklearn digits dataset and prepare it for training.

    The dataset contains 1,797 8x8 grayscale images of handwritten
    digits (0-9). Each pixel value is 0-16.

    Parameters
    ----------
    test_size : float
        Fraction of data to reserve for testing
    seed : int
        Random seed for reproducible splits

    Returns
    -------
    X_train, X_test : ndarray
        Normalized input features (pixel values scaled to 0-1)
    y_train, y_test : ndarray
        One-hot encoded target labels
    y_train_labels, y_test_labels : ndarray
        Integer labels (0-9) for evaluation
    """
    digits = load_digits()
    X = digits.data          # shape: (1797, 64)
    y_labels = digits.target  # shape: (1797,) values 0-9

    # Normalize pixel values to [0, 1]
    X = X / 16.0

    # One-hot encode labels: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    n_classes = 10
    y_onehot = np.zeros((len(y_labels), n_classes))
    y_onehot[np.arange(len(y_labels)), y_labels] = 1.0

    # Train/test split
    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = \
        train_test_split(X, y_onehot, y_labels, test_size=test_size,
                         random_state=seed, stratify=y_labels)

    return X_train, X_test, y_train, y_test, y_train_labels, y_test_labels


# =============================================================================
# DIGIT CLASSIFIER NETWORK
# =============================================================================

class DigitNetwork:
    """
    A multi-layer neural network for digit classification.

    Enhancements over the basic NeuralNetwork in file 05:
      - Xavier initialization (better starting weights)
      - Softmax output layer (proper probability distribution)
      - Cross-entropy loss (better for classification)
      - Mini-batch training (faster convergence)

    Parameters
    ----------
    layer_sizes : list of int
        e.g. [64, 32, 16, 10] for our digit classifier
    seed : int
        Random seed for reproducibility
    """

    def __init__(self, layer_sizes, seed=42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        # Xavier initialization: scale weights by sqrt(2 / fan_in)
        # This keeps signals from exploding or vanishing as they
        # flow through the network
        self.weights = []
        self.biases = []
        for i in range(self.n_layers - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        """
        Forward pass through the network.

        Hidden layers use sigmoid activation.
        Output layer uses softmax (for probability distribution).
        """
        self.z_values = []
        self.a_values = [X]

        current = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ w + b
            self.z_values.append(z)

            if i < len(self.weights) - 1:
                # Hidden layer: sigmoid
                a = sigmoid(z)
            else:
                # Output layer: softmax
                a = softmax(z)

            self.a_values.append(a)
            current = a

        return current

    def backward(self, y):
        """
        Backward pass using cross-entropy loss with softmax output.

        For softmax + cross-entropy, the gradient at the output simplifies to:
            delta = prediction - target

        This elegant simplification is one reason this combination is so popular.
        """
        n = len(y)
        dw_list = [None] * (self.n_layers - 1)
        db_list = [None] * (self.n_layers - 1)

        # Output layer gradient (softmax + cross-entropy)
        # This simplification is mathematically exact
        delta = (self.a_values[-1] - y) / n

        # Work backward through layers
        for layer in range(self.n_layers - 2, -1, -1):
            a_prev = self.a_values[layer]
            dw_list[layer] = a_prev.T @ delta
            db_list[layer] = np.sum(delta, axis=0)

            if layer > 0:
                # Propagate delta to previous layer
                delta = (delta @ self.weights[layer].T) * \
                        sigmoid_derivative(self.a_values[layer])

        return dw_list, db_list

    def compute_loss(self, predictions, targets):
        """
        Cross-entropy loss for classification.

        Measures how well the predicted probability distribution
        matches the true distribution (one-hot target).
        """
        # Clip to avoid log(0)
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        return -np.mean(np.sum(targets * np.log(predictions), axis=1))

    def train(self, X_train, y_train, X_test=None, y_test=None,
              learning_rate=0.5, n_epochs=100, batch_size=32, verbose=True):
        """
        Train with mini-batch gradient descent.

        Parameters
        ----------
        X_train, y_train : ndarray
            Training data and one-hot labels
        X_test, y_test : ndarray, optional
            Test data for monitoring generalization
        learning_rate : float
            Step size
        n_epochs : int
            Number of passes through the training data
        batch_size : int
            Number of samples per mini-batch
        verbose : bool
            Print progress

        Returns
        -------
        history : dict
            Training and test metrics over time
        """
        n_samples = len(X_train)
        history = {
            'train_losses': [],
            'test_losses': [],
            'train_accuracies': [],
            'test_accuracies': [],
        }

        for epoch in range(n_epochs):
            # Shuffle training data each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward + backward + update
                self.forward(X_batch)
                dw_list, db_list = self.backward(y_batch)

                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * dw_list[i]
                    self.biases[i] -= learning_rate * db_list[i]

            # Evaluate on full training set
            train_pred = self.forward(X_train)
            train_loss = self.compute_loss(train_pred, y_train)
            train_acc = np.mean(np.argmax(train_pred, axis=1) ==
                                np.argmax(y_train, axis=1)) * 100
            history['train_losses'].append(train_loss)
            history['train_accuracies'].append(train_acc)

            # Evaluate on test set if available
            if X_test is not None:
                test_pred = self.forward(X_test)
                test_loss = self.compute_loss(test_pred, y_test)
                test_acc = np.mean(np.argmax(test_pred, axis=1) ==
                                   np.argmax(y_test, axis=1)) * 100
                history['test_losses'].append(test_loss)
                history['test_accuracies'].append(test_acc)

            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                msg = (f"    Epoch {epoch:3d}: "
                       f"train_loss={train_loss:.4f}, "
                       f"train_acc={train_acc:.1f}%")
                if X_test is not None:
                    msg += f", test_acc={test_acc:.1f}%"
                print(msg)

        return history

    def predict(self, X):
        """Return predicted class labels (0-9)."""
        output = self.forward(X)
        return np.argmax(output, axis=1)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_sample_digits(X, y_labels):
    """Plot a 2x5 grid of example digits from the dataset."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    fig.suptitle("Sample Handwritten Digits (8x8 pixels)",
                 fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].reshape(8, 8), cmap='gray_r', interpolation='nearest')
        ax.set_title(f"Label: {y_labels[i]}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_curves(history):
    """Plot loss and accuracy curves during training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Progress",
                 fontsize=14, fontweight='bold')

    # Loss
    ax1.plot(history['train_losses'], 'b-', label='Train', linewidth=1.5)
    if history['test_losses']:
        ax1.plot(history['test_losses'], 'r-', label='Test', linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history['train_accuracies'], 'b-', label='Train', linewidth=1.5)
    if history['test_accuracies']:
        ax2.plot(history['test_accuracies'], 'r-', label='Test', linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """Plot which digits get confused with each other."""
    n_classes = 10
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("Confusion Matrix: Which Digits Get Confused?",
                 fontsize=14, fontweight='bold')

    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color=color, fontsize=9)

    ax.set_xlabel("Predicted Digit")
    ax.set_ylabel("True Digit")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_title("Rows = true label, Columns = predicted label")

    plt.tight_layout()
    plt.show()


def plot_predictions(X_test, y_true, y_pred, n_show=20):
    """
    Show test images with predicted and true labels.

    Correct predictions in green, wrong ones in red.
    """
    n_cols = 5
    n_rows = n_show // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2 * n_rows))
    fig.suptitle("Predictions on Test Data (Green=Correct, Red=Wrong)",
                 fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i >= n_show or i >= len(X_test):
            ax.axis('off')
            continue
        ax.imshow(X_test[i].reshape(8, 8), cmap='gray_r',
                  interpolation='nearest')
        correct = y_pred[i] == y_true[i]
        color = 'green' if correct else 'red'
        ax.set_title(f"Pred:{y_pred[i]} True:{y_true[i]}",
                     fontsize=9, color=color, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_what_the_network_learned(net):
    """
    Visualize first-layer weights reshaped as 8x8 images.

    Each hidden neuron in the first layer has 64 weights -- one per pixel.
    Reshaping these weights to 8x8 shows what "feature" each neuron
    has learned to detect.
    """
    W1 = net.weights[0]  # shape: (64, 32)
    n_show = min(16, W1.shape[1])
    n_cols = 8
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    fig.suptitle("What the Network Learned: First-Layer Weights as 8x8 Images",
                 fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i < n_show:
            weights_img = W1[:, i].reshape(8, 8)
            ax.imshow(weights_img, cmap='RdBu_r', interpolation='nearest')
            ax.set_title(f"Neuron {i}", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    """Run the digit classifier demonstration."""
    print("=" * 70)
    print("PUTTING IT ALL TOGETHER: HANDWRITTEN DIGIT RECOGNITION")
    print("=" * 70)

    # --- Load Data ---
    print("\n--- LOADING DATA ---")
    print("""
The dataset: 1,797 handwritten digits (0-9), each an 8x8 grayscale image.
Each pixel is a number 0-16, which we normalize to 0-1.
That gives us 64 input features (one per pixel).
""")
    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = \
        load_and_prepare_data()

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")
    print(f"  Input features:   {X_train.shape[1]} (8x8 pixels)")
    print(f"  Output classes:   10 (digits 0-9)")

    # --- Show sample digits ---
    print("\n--- SAMPLE DIGITS ---")
    print("Plotting example digits from the dataset...")
    plot_sample_digits(X_train, y_train_labels)

    # --- Build Network ---
    print("\n--- BUILDING THE NETWORK ---")
    print("""
Architecture: 64 -> 32 -> 16 -> 10
  - 64 inputs:     one per pixel
  - 32 hidden:     first hidden layer (learns low-level features)
  - 16 hidden:     second hidden layer (learns higher-level features)
  - 10 outputs:    one per digit class (softmax probabilities)

Total parameters: 64*32 + 32 + 32*16 + 16 + 16*10 + 10 = 2,738
(That's all it takes to recognize handwriting!)
""")
    net = DigitNetwork([64, 32, 16, 10], seed=42)

    # Test before training
    initial_pred = net.predict(X_test)
    initial_acc = np.mean(initial_pred == y_test_labels) * 100
    print(f"  Accuracy before training: {initial_acc:.1f}% (random guessing ~10%)")

    # --- Train ---
    print("\n--- TRAINING ---")
    print("  Training with mini-batch gradient descent...")
    print("  Watch the accuracy climb from ~10% toward ~95%!\n")

    history = net.train(X_train, y_train, X_test, y_test,
                        learning_rate=0.5, n_epochs=150, batch_size=32,
                        verbose=True)

    # --- Final Evaluation ---
    print("\n--- FINAL EVALUATION ---")
    y_pred = net.predict(X_test)
    final_acc = np.mean(y_pred == y_test_labels) * 100
    print(f"\n  Final test accuracy: {final_acc:.1f}%")
    print(f"  That's {int(final_acc * len(y_test_labels) / 100)} out of "
          f"{len(y_test_labels)} digits correctly classified!")

    # Per-digit accuracy
    print("\n  Per-digit accuracy:")
    for digit in range(10):
        mask = y_test_labels == digit
        digit_acc = np.mean(y_pred[mask] == digit) * 100
        print(f"    Digit {digit}: {digit_acc:.1f}%")

    # --- Visualizations ---
    print("\n--- VISUALIZATION: TRAINING CURVES ---")
    plot_training_curves(history)

    print("\n--- VISUALIZATION: CONFUSION MATRIX ---")
    print("Which digits does the network confuse?")
    plot_confusion_matrix(y_test_labels, y_pred)

    print("\n--- VISUALIZATION: PREDICTIONS ---")
    print("Showing predictions on test images...")
    plot_predictions(X_test, y_test_labels, y_pred, n_show=20)

    print("\n--- VISUALIZATION: WHAT THE NETWORK LEARNED ---")
    print("First-layer weights reshaped as 8x8 images show discovered features...")
    plot_what_the_network_learned(net)

    # --- Reflection ---
    print("\n" + "=" * 70)
    print("REFLECTION")
    print("=" * 70)
    print(f"""
We built a neural network that recognizes handwritten digits with
{final_acc:.1f}% accuracy, using ONLY:
  - Addition and multiplication (forward pass)
  - The chain rule (backpropagation)
  - Gradient descent (learning)

No hand-written rules about what "3" looks like.
No explicit programming of digit features.
The network discovered its own representations from data alone.

This is the Bitter Lesson in action:
  "General methods that leverage computation are ultimately the most
   effective, and by a large margin." - Rich Sutton

And it connects to emergence:
  Simple components (neurons doing weighted sums and sigmoids) combine
  to produce complex behavior (recognizing handwriting) that no single
  component could achieve alone.

What we built today is the same PRINCIPLE behind systems like:
  - GPT (language): same math, billions more parameters, text data
  - DALL-E (images): same math, different architecture, image data
  - AlphaFold (proteins): same math, specialized architecture, protein data

The difference is scale and architecture -- not the fundamental ideas.
Everything we covered today IS the foundation of modern AI.
""")


if __name__ == "__main__":
    main()
