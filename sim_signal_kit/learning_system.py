import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import count
from neurodsp.burst import detect_bursts_dual_threshold


class Approximator(nn.Module):
    """
    The Approximator learns to estimate the optimal parameter set θ for a given signal x,
    such that when passed into a pre-existing model M, the model output M(θ, x)
    closely matches the expected output y.

    ## **Mathematical Formulation**
    Given:
        - A model **M(θ, x) → y**, where:
            - `x` ∈ ℝ^d is an input signal
            - `θ` ∈ ℝ^p is a set of tunable parameters
            - `y` ∈ ℝ^k is the desired model output
        - A dataset **D = {(xᵢ, yᵢ)}** where `yᵢ` represents the expected output

    When trained, the Approximator is a prediction function A(x) that estimates the optimal parameters θ for a given signal x.
    The goal is to run A inside dualthresh so that Dualthresh(x, A(x)) ≈ ground truth.

    We M(A(x), x) ≈ y, as optimized by scipy_optimize.minimize with the model.

    We train neural network A(x; ϕ). x is the signal and ϕ are the params. They are passed into separate NNs.
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
        torch.autograd.set_detect_anomaly(True)
        super().__init__()
        self.signal_dim = signal_dim
        self.param_dim = param_dim

        self.sig_to_param_net = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim),
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
        # Create a copy and modify only the required parameter
        param_pred_copy = param_pred.clone()
        param_pred_copy[:, 0] = torch.nn.functional.softplus(param_pred[:, 0]) + 2  # Ensure output > 2

        return param_pred_copy
    
    def predict(self, signal):
        """
        Given an input signal x, predict the optimal parameters θ and compute M(θ, x).

        Args:
        - signal (ndarray or Tensor): Input signal

        Returns:
        - y_pred (Tensor): Model params θ, where M(θ, x) ≈ y. y is the burst selection here.
        """
        self.eval()
        with torch.no_grad():
            if isinstance(signal, np.ndarray):
                signal = torch.tensor(signal, dtype=torch.float32)

            param_prediction = self.sig_to_param_net(signal)
            #         # Create a copy and modify only the required parameter
            # param_pred_copy = param_prediction.clone()
            # param_pred_copy[:, 0] = torch.nn.functional.softplus(param_prediction[:, 0]) + 2  # Ensure output > 2

            return param_prediction

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

        for epoch in range(num_epochs):
        # for epoch in count(0):
            total_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(
                    device
                )  # Move to device
                self.optimizer.zero_grad()
                predictions = self.forward(inputs)  # Forward pass
                loss = self.criterion(predictions, targets)  # Compute loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{max(num_epochs, epoch)}, Loss: {avg_loss:.6f}")
            if epoch >= loss_list.shape[0]:
                loss_list = np.append(loss_list, avg_loss)
            else:
                loss_list[epoch] = avg_loss
        print("Training complete.")
        print("Final loss:", loss_list[-1])

        self.eval()  # Switch to evaluation mode

def mse_dualthresh(signals, y_pred, y_true, fs=1000):
    """
    Loss function for dual threshold optimization. 
    """
    total_loss = 0
     

    for i in range(len(signals)):

        yp0 = y_pred[i][0]
        yp1 = y_pred[i][1]

        yp0 = max(yp0, 2)
        yp1 = max(yp1, 0)
        is_burst = detect_bursts_dual_threshold(signals[i],
                                                 fs, (max(9,y_pred[i][0]),
                                                      max(9,y_pred[i][0])+y_pred[i][1]),
                                                      (y_pred[i][2], y_pred[i][2]+y_pred[i][3])
                                                      )
        # Compute loss by using the dualthresh model to predict the intervals
        # then set the error as the difference between the predicted intervals and the ground truth
        if is_burst is None:
            return 1e99
        if len(is_burst) == 0:
            return 1e99
        
        for j in range(len(is_burst)):
            
            bounds_list=[0,0]
            found_onset = False
            found_offset = False
            if not found_onset and is_burst[j]==1:
                bounds_list[0] = j
                found_onset = True
            if found_onset and not found_offset and is_burst[j]==0:
                bounds_list[1] = j
            if found_onset and found_offset:
                break
        interval = np.array(bounds_list)

        loss = (interval[0] - y_true[i][0]) ** 2 + (interval[1] - y_true[i][1]) ** 2
        total_loss+=loss

    return total_loss