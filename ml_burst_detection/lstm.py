# conv_transformer_approximator.py

# ltsm implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.onnx import export


# ---------------------------
# Loss: interval-aware (Huber + 1-IoU) with no-burst support
# ---------------------------
def interval_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    delta_huber: float = 5.0,
    w_endpoints: float = 1.0,
    w_overlap: float = 1.0,
    w_noburst_len: float = 5.0,
    w_noburst_anchor: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Interval-aware loss for [onset, offset] regression.

    Conventions:
      - Positive (burst present): y_true = [onset, offset], onset < offset
      - No-burst: y_true = [-1, -1]

    For positives:
      loss = Huber([onset, offset]) + (1 - IoU(pred, true))

    For no-burst:
      loss = w_noburst_len * length_pred + w_noburst_anchor * (|onset+1| + |offset+1|)
             where length_pred = relu(offset - onset)

    Args:
        y_pred: (B, 2) predicted [onset, offset]
        y_true: (B, 2) ground-truth [onset, offset] or [-1, -1] for no-burst
    """
    onset_p,  offset_p  = y_pred[:, 0], y_pred[:, 1]
    onset_t,  offset_t  = y_true[:, 0], y_true[:, 1]

    # Masks
    noburst = (onset_t < 0) & (offset_t < 0)  # [-1, -1]
    posmask = ~noburst

    loss = 0.0

    # ---- Positive intervals ----
    if posmask.any():
        op = onset_p[posmask]; of = offset_p[posmask]
        ot = onset_t[posmask]; ft = offset_t[posmask]

        # Endpoint Huber
        end_pred = torch.stack([op, of], dim=1)
        end_true = torch.stack([ot, ft], dim=1)
        hub = F.huber_loss(end_pred, end_true, delta=delta_huber, reduction="none").sum(dim=1)

        # IoU on 1D intervals (differentiable via ReLUs)
        inter = torch.relu(torch.min(of, ft) - torch.max(op, ot))
        union = torch.relu(torch.max(of, ft) - torch.min(op, ot)) + eps
        iou = inter / union
        overlap_loss = 1.0 - iou

        loss_pos = w_endpoints * hub + w_overlap * overlap_loss
        loss = loss + loss_pos.mean()

    # ---- No-burst intervals ----
    if noburst.any():
        op = onset_p[noburst]; of = offset_p[noburst]
        length = torch.relu(of - op)  # predicted length (>= 0)
        anchor = -1.0
        loc = torch.abs(op - anchor) + torch.abs(of - anchor)
        loss_nb = w_noburst_len * length + w_noburst_anchor * loc
        loss = loss + loss_nb.mean()

    # Return as tensor
    if not isinstance(loss, torch.Tensor):
        loss = torch.tensor(loss, device=y_pred.device, dtype=y_pred.dtype)
    return loss


# ---------------------------
# Positional encoding (sin-cos)
# ---------------------------
class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, D)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # not a learnable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# ---------------------------
# Model: Conv + Transformer (or Transformer + Conv) + Valid-Interval Head
# ---------------------------
class ConvTransformerApproximator(nn.Module):
    """
    Maps a 1D signal (B, T) -> (B, 2) [onset, offset].

    Order:
      - conv_first=True: Conv -> Linear(to d_model) -> PosEnc -> Transformer -> Pool -> Head
      - conv_first=False: Linear(1->d_model) -> PosEnc -> Transformer -> Linear(to conv_dim) -> Conv -> Pool -> Head

    Valid interval parameterization:
      raw -> onset, delta
      offset = onset + softplus(delta)    # ensures offset >= onset
    """

    def __init__(
        self,
        signal_dim: int,        # sequence length T
        d_model: int = 256,     # Transformer hidden size
        nhead: int = 8,
        num_layers: int = 4,
        conv_channels: int = 128,
        conv_first: bool = True,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.signal_dim = signal_dim
        self.conv_first = conv_first

        # ---- Convolutional block ----
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=7, padding=3),
            nn.GELU(),
        )

        # ---- Linear projections ----
        # For conv_first branch: (B,T,Cc) -> (B,T,D)
        self.to_d_model_from_conv = nn.Linear(conv_channels, d_model)
        # For transformer_first branch: (B,T,1) -> (B,T,D)
        self.to_d_model_from_raw = nn.Linear(1, d_model)
        # Transformer_first: back to conv channels for local refinement
        self.to_conv_dim = nn.Linear(d_model, conv_channels)

        # ---- Transformer encoder ----
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = SinusoidalPosEnc(d_model)

        # ---- Head: high -> low reduction ----
        # Global avg + max pooling across time, concat → MLP → 2 (onset_raw, delta_raw)
        red_in = d_model if conv_first else conv_channels
        self.head = nn.Sequential(
            nn.Linear(red_in * 2, red_in),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(red_in, 2),  # [onset_raw, delta_raw]
        )

        # ---- Training bits ----
        self.criterion = interval_loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def _pool_reduce(self, seq_feats: torch.Tensor) -> torch.Tensor:
        """Global average + max pooling over time: (B, T, C) -> (B, 2C)."""
        avg = seq_feats.mean(dim=1)
        mx  = seq_feats.max(dim=1).values
        return torch.cat([avg, mx], dim=-1)

    def _valid_interval_head(self, rep: torch.Tensor) -> torch.Tensor:
        """Map pooled representation to a valid [onset, offset]."""
        raw = self.head(rep)                # (B, 2)
        onset = raw[:, 0]
        delta = F.softplus(raw[:, 1]) + 1e-3  # tiny floor for stability
        offset = onset + delta
        return torch.stack([onset, offset], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) float
        returns: (B, 2) [onset, offset]
        """
        if self.conv_first:
            # Conv expects (B, C=1, T)
            f = self.conv(x.unsqueeze(1))      # (B, Cc, T)
            f = f.transpose(1, 2)              # (B, T, Cc)
            f = self.to_d_model_from_conv(f)   # (B, T, D)
            f = self.pos(f)                    # add positional encoding
            f = self.transformer(f)            # (B, T, D)
            rep = self._pool_reduce(f)         # (B, 2D)
            out = self._valid_interval_head(rep)
            return out
        else:
            # Transformer first on raw signal
            f = x.unsqueeze(-1)                         # (B, T, 1)
            f = self.to_d_model_from_raw(f)            # (B, T, D)
            f = self.pos(f)                            # (B, T, D) + PE
            f = self.transformer(f)                    # (B, T, D)
            # Local refinement via conv
            f = self.to_conv_dim(f).transpose(1, 2)    # (B, Cc, T)
            f = self.conv(f)                           # (B, Cc, T)
            f = f.transpose(1, 2)                      # (B, T, Cc)
            rep = self._pool_reduce(f)                 # (B, 2Cc)
            out = self._valid_interval_head(rep)
            return out

    @torch.no_grad()
    def predict(self, signal: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        signal: (T,) or (B, T) -> returns (B, 2) on the same device as the model
        """
        self.eval()
        if isinstance(signal, np.ndarray):
            signal = torch.tensor(signal, dtype=torch.float32)
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        device = next(self.parameters()).device
        return self.forward(signal.to(device))

    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 10,
        device: str = "cpu",
        plot: bool = True,
    ):
        """
        Train the model with interval_loss. Logs (log10 scale) are returned.
        """
        self.to(device)
        train_log, val_log = [], []
        best_loss, best_state = float("inf"), None

        for epoch in range(num_epochs):
            # ---- Train ----
            self.train()
            total = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                self.optimizer.zero_grad()
                pred = self.forward(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                total += loss.item()
            tr_loss = total / max(1, len(train_loader))
            train_log.append(np.log10(tr_loss + 1e-8))

            # ---- Validate ----
            self.eval()
            vtotal = 0.0
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv, yv = xv.to(device), yv.to(device)
                    vpred = self.forward(xv)
                    vloss = self.criterion(vpred, yv)
                    vtotal += vloss.item()
            va_loss = vtotal / max(1, len(val_loader))
            val_log.append(np.log10(va_loss + 1e-8))

            # ---- Bookkeeping ----
            if va_loss < best_loss:
                best_loss, best_state = va_loss, self.state_dict()

            self.scheduler.step(va_loss)
            print(f"Epoch {epoch+1}/{num_epochs} | Train: {tr_loss:.6f} | Val: {va_loss:.6f}")

        if best_state is not None:
            self.load_state_dict(best_state)
            print("Loaded best validation checkpoint.")

        if plot:
            plt.figure()
            plt.plot(train_log, label="Train LogLoss")
            plt.plot(val_log, label="Val LogLoss")
            plt.xlabel("Epoch")
            plt.ylabel("log10(loss)")
            plt.legend()
            plt.grid(True)
            plt.title("Training / Validation Loss")
            plt.show()

        # Export ONNX (dummy single example)
        dummy = torch.randn(1, self.signal_dim, device=next(self.parameters()).device)
        export(self, dummy, "conv_transformer_approximator.onnx", verbose=False)
        return train_log, val_log


# Optional: plain MSE helper (not used)
def mse_loss(y_pred, y_true, reduction="mean"):
    return F.mse_loss(y_pred, y_true, reduction=reduction)