# ps-neuralnetwork

A minimal neural network framework built from scratch in pure Python, featuring a custom automatic-differentiation `Value` class, multi-layer perceptron (`Neural_Network`), training loop, and decision boundary visualization. Perfect for educational purposes and quick experimentation with 2D classification tasks.

---

## Features

- **Automatic Differentiation** via a custom `Value` class  
- **Multi-Layer Perceptron** with configurable hidden layers and tanh activations  
- **Training Utility** with Mean Squared Error loss and manual gradient descent  
- **Decision Boundary Visualization** for 2D datasets using Matplotlib  

---

## Installation

```bash
git clone https://github.com/PritiyaxShukla/ps_neuralnetwork.git
cd ps_neuralnetwork
pip install ps_neuralnetwork .
```

This installs the package in “editable” mode so you can modify the source and immediately test changes.

---

## Quick Start

```python
# 1. Import training helper and network
from ps_neuralnetwork.training import training
from ps_neuralnetwork.nn_scratch import Neural_Network

# 2. Generate 2D classification data
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1  # Convert labels from {0,1} to {-1,+1}

# 3. Visualize raw data
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')
plt.title("Make Moons Dataset")
plt.show()

# 4. Create and configure Neural Network
net = Neural_Network(input_size=2, layer_sizes=[4, 3, 5, 2, 1])

# 5. Train the network
learning_rate = 0.001
num_epochs     = 500
trainer = training(net, X, y, learning_rate, num_epochs)

# Run training
trainer()

# 6. Plot training progress and decision boundary
trainer.plot(smoothness=0.05)
```

---

## API Reference

### `ps_neuralnetwork.nn_scratch.Neural_Network(input_size, layer_sizes)`

- **`input_size`**: Number of input features (e.g., 2 for 2D data)  
- **`layer_sizes`**: List of neuron counts for each hidden and output layer (e.g., `[4,3,1]`)

Creates a feed-forward network with tanh activations.

---

### `ps_neuralnetwork.training.training(net, X, y, learning_rate, num_epochs)`

- **`net`**: Instance of `Neural_Network`  
- **`X`**: Feature array of shape `(n_samples, input_size)`  
- **`y`**: Label array of shape `(n_samples,)` with values `-1` or `+1`  
- **`learning_rate`**: Step size for gradient descent (e.g., `0.001`)  
- **`num_epochs`**: Number of training epochs (e.g., `500`)

Returns a **trainer** object with:

- **Callable**: `trainer()` to start training  
- **`trainer.plot(smoothness)`**: Visualizes decision boundary and loss over epochs  
  - **`smoothness`** controls contour resolution  

---

## Package Structure

```
ps-neuralnetwork/
├── src/
│   └── ps_neuralnetwork/
│       ├── __init__.py
│       ├── value.py        # Automatic differentiation engine
│       ├── network.py      # Multi-layer perceptron
│       └── training.py     # Training loop and visualization
├── README.md              # This file
├── LICENSE                # MIT License
└── pyproject.toml         # Build and metadata
```

---

## License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Built with NumPy, Matplotlib, and scikit-learn  
- Inspired by “neural networks from scratch” tutorials and micro-autograd implementations  


## The Value class was adapted from Andrej Karpathy’s micrograd project (https://github.com/karpathy/micrograd), MIT License.

Happy experimenting! 🎉