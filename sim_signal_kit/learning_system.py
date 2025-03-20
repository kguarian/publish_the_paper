import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import count

class Approximator(nn.Module):
    """
    The Approximator learns to estimate the optimal parameter set for a given signal x,
    such that when passed into a pre-existing model M like dualthresh, the model output M(params, x) 
    closely matches the expected output y.

    ## Mathematical Formulation
    Given:
        - A model M(params, x) → y, where:
            - `x` ∈ ℝ^d is an input signal
            - `params` ∈ ℝ^p is a set of tunable parameters
            - `y` ∈ ℝ^k is the desired model output
        - A dataset D = {(xᵢ, yᵢ)} where `yᵢ` is the ground truth params.

    When trained, the Approximator is a prediction function A(x) that estimates the optimal params for a given signal x.
    The goal is to run A inside dualthresh so that Dualthresh(x, A(x)) ≈ ground truth.
    
    We M(A(x), x) ≈ y, as optimized by scipy_optimize.minimize with the model.

    We train neural network A(x; params). x is the signal.
    The loss function is the mean squared error (MSE) between the Approximator's param guess and the optimal params.

    
    """

    def __init__(self, signal_dim, param_dim, hidden_dim=128, lr=0.001):
        """
        Initializes the Approximator.

        Args:
        - model: The pre-existing model M(θ, x) -> y.
        - signal_dim: Dimension of the input signal x.
        - param_dim: Dimension of the estimated parameter space θ.
        - prediction_wrapper: Function to call model M with estimated parameters.
        - hidden_dim: Hidden layer size for neural network.
        - lr: Learning rate for optimizer.
        """
        super().__init__()
        self.signal_dim = signal_dim
        self.param_dim = param_dim

        self.sig_to_param_net = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim)
        )
        # Optimizer and loss function
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        """
        Forward pass: Given a signal, estimate parameters θ.
        
        Args:
        - x (Tensor): Input signal of shape (batch_size, signal_dim)

        Returns:
        - model_output (Tensor): Output of M(θ, x)
        """
        param_pred = self.sig_to_param_net(x)
        return param_pred

    def predict(self, signal):
        """
        Given an input signal x, predict the optimal parameters θ and compute M(θ, x).

        Args:
        - signal (ndarray or Tensor): Input signal

        Returns:
        - y_pred (ndarray): Predicted output M(A(x), x)
        """
        self.eval()
        with torch.no_grad():
            if isinstance(signal, np.ndarray):
                signal = torch.tensor(signal, dtype=torch.float32)
            
            theta_pred = self.sig_to_param_net(signal)
            return theta_pred.cpu().numpy()

    def train_model(self, train_loader, num_epochs=10, device="mps"):
        """
        Train the approximator to optimize parameters θ.

        Args:
        - train_loader: DataLoader providing (input signals, expected outputs)
        - num_epochs: Number of training iterations
        - device: Device to use for training (e.g., "cuda", "mps", "cpu")
        """
        self.train()  # Set the model to training mode
        self.to(device)  # Ensure model is on the correct device

        loss_list = np.zeros(num_epochs)
        def condition(epoch):
            if epoch<=num_epochs:
                return True
            return np.mean(loss_list[-int(num_epochs/4):])<np.mean(loss_list[-int(num_epochs/2):])
        # for epoch in range(num_epochs):
        for epoch in count(0):
            if not condition(epoch):
                break
            total_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move to device

                self.optimizer.zero_grad()
                predictions = self.forward(inputs)  # Forward pass
                loss = self.criterion(predictions, targets)  # Compute loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{max(num_epochs, epoch)}, Loss: {avg_loss:.6f}")
            if epoch>=loss_list.shape[0]:
                loss_list = np.append(loss_list, avg_loss)
            else:
                loss_list[epoch] = avg_loss
        print("Training complete.")
        print("Final loss:", loss_list[-1])

        self.eval()  # Switch to evaluation mode