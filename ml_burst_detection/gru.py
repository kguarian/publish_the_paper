import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.onnx import export

# gru implementation

def interval_loss(y_pred: torch.Tensor, y_true: torch.Tensor, penalty: float = 1e6):
    """
    Custom loss for onset/offset predictions.

    Args:
        y_pred: predicted intervals, shape (B, 2) with [onset, offset]
        y_true: ground truth intervals, shape (B, 2)
        penalty: penalty value for invalid intervals (offset < onset)

    Returns:
        Scalar loss value
    """
    # Standard MSE between predictions and truth
    mse = F.mse_loss(y_pred, y_true, reduction="mean")

    # Extract onsets and offsets
    onset_pred, offset_pred = y_pred[:, 0], y_pred[:, 1]

    # Invalid case: offset < onset
    invalid_mask = (offset_pred < onset_pred) | (onset_pred > offset_pred)

    if invalid_mask.any():
        # Add penalty (scaled by number of invalids)
        mse = mse + penalty * invalid_mask.float().mean()

    return mse

class Approximator(nn.Module):
    """
    GRU Approximator that directly maps a 1D signal to a set of burst detection parameters.

    - Input: signal x ∈ ℝ^signal_dim
    - Output: parameters θ ∈ ℝ^param_dim
    """

    def __init__(self, signal_dim: int, param_dim: int, hidden_dim: int = 64, lr: float = 1e-3):
        super().__init__()
        self.signal_dim = signal_dim
        self.param_dim = param_dim

        self.rnn = nn.GRU(input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, param_dim),
        )

        # self.criterion = nn.HuberLoss()
        self.criterion = interval_loss
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RNN.

        Args:
            x: Input tensor of shape (batch_size, signal_dim)

        Returns:
            Output tensor of shape (batch_size, param_dim)
        """
        x = x.unsqueeze(-1)  # (B, T) -> (B, T, 1)
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]  # (B, H)
        return self.fc(last_hidden)

    def predict(self, signal: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Predict parameters θ from a signal.

        Args:
            signal: (signal_dim,) or (batch_size, signal_dim)

        Returns:
            Predicted θ values
        """
        self.eval()
        with torch.no_grad():
            if isinstance(signal, np.ndarray):
                signal = torch.tensor(signal, dtype=torch.float32)
            if signal.ndim == 1:
                signal = signal.unsqueeze(0)  # Add batch
            signal = signal.to(next(self.parameters()).device)
            return self.forward(signal)

    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 10,
        device: str = "cpu",
        plot: bool = True,
    ):
        """
        Train the RNN-based parameter predictor.

        Args:
            train_loader: Dataloader for training
            val_loader: Dataloader for validation
            num_epochs: Number of epochs
            device: torch device ("cpu", "cuda", "mps")
            plot: Whether to plot loss curves
        """
        self.to(device)
        train_log, val_log = [], []
        best_loss = float("inf")
        best_weights = None

        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0.0

            for i, (x_batch, y_batch) in enumerate(train_loader):
                print(f"Training batch {i+1}/{len(train_loader)}", end="\r")
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                self.optimizer.zero_grad()
                pred = self.forward(x_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            print(len(train_loader))

            avg_train_loss = total_train_loss / len(train_loader)
            train_log.append(np.log10(avg_train_loss + 1e-8))

            # Validation
            self.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for i, (x_val, y_val) in enumerate(val_loader):
                    print(f"Processing batch {i+1}/{len(val_loader)}", end="\r")
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    val_pred = self.forward(x_val)
                    val_loss = self.criterion(val_pred, y_val)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_log.append(np.log10(avg_val_loss + 1e-8))

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_weights = self.state_dict()

            self.scheduler.step(avg_val_loss)
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if best_weights is not None:
            self.load_state_dict(best_weights)

        print("Training complete. Best val loss:", best_loss)

        if plot:
            plt.figure()
            plt.plot(train_log, label="Train LogLoss")
            plt.plot(val_log, label="Val LogLoss")
            plt.xlabel("Epochs")
            plt.ylabel("Log10(MSE Loss)")
            plt.legend()
            plt.title("Training and Validation Loss")
            plt.grid(True)
            plt.show()

        # Export model
        dummy_input = torch.randn(1, self.signal_dim).to(next(self.parameters()).device)
        export(self, dummy_input, "approximator_gru.onnx", verbose=False)

        return train_log, val_log


def mse_loss(y_pred, y_true):
    """
    Compute mean squared error between predictions and ground truth.
    """
    return F.mse_loss(y_pred, y_true, reduction="none") 