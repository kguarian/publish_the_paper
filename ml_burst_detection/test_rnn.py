import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

from rnn import Approximator
from sims_direct import generate_training_data_approximator, fetch_val_data, fs
from neurodsp.burst import detect_bursts_dual_threshold


def run_on_signal():
    """Train Approximator to predict burst intervals directly from signals."""

    # Pick device: Apple MPS, CUDA, or CPU
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    # Dataset sizes
    train_size, test_size = 100, 10

    # Generate data
    x_test, y_test, gt_y_test = generate_training_data_approximator(n_seconds = 2, n_sims=test_size, mode="test", pos_whole_ratio=1.0)
    x_train, y_train, gt_y_train = generate_training_data_approximator(n_seconds = 2, n_sims = train_size, mode="train", pos_whole_ratio=1.0)
    x_val, y_val, gt_y_val = fetch_val_data()

    # Use ground truth bounds as targets
    y_train, y_test, y_val = gt_y_train, gt_y_test, gt_y_val

    # Convert to tensors
    def to_tensor(x): return torch.from_numpy(np.array(x)).float().to(device)
    x_train, y_train = to_tensor(x_train), to_tensor(y_train)
    x_val, y_val = to_tensor(x_val), to_tensor(y_val)
    x_test, y_test = to_tensor(x_test), to_tensor(y_test)

    # Shapes
    input_shape, output_shape = x_train.shape[1], y_train.shape[1]

    # Datasets
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=1, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=1, shuffle=True)

    # Model
    model = Approximator(signal_dim=input_shape,
                         param_dim=output_shape,
                         hidden_dim=512,
                         lr=0.003).to(device)

    # Train
    model.train_model(train_loader, val_loader=val_loader, num_epochs=10)

    # Predict
    y_pred = model.predict(x_test).cpu().numpy()
    # y_true = y_test.cpu().numpy()
    y_true = gt_y_test

    # Eval
    mse = np.mean((y_pred - y_true) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")

    # Plot nicely
    ncols = 2
    nrows = int(np.ceil(test_size / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 2), sharex=False, sharey=False)
    axes = axes.flatten()

    for i in range(test_size):
        ax = axes[i]
        pred_on, pred_off = y_pred[i]
        true_on, true_off = y_true[i]

        # Run dual-threshold for comparison
        is_burst = detect_bursts_dual_threshold(x_test[i].cpu().numpy(),
                                                fs, dual_thresh=[1, 2], f_range=[5, 10])
        # First interval
        bounds = [0, 0]
        found_onset = False
        for j, flag in enumerate(is_burst):
            if not found_onset and flag == 1:
                bounds[0] = j
                found_onset = True
            elif found_onset and flag == 0:
                bounds[1] = j
                break

        # Plot signal + intervals
        ax.plot(x_test[i].cpu().numpy(), label="signal", lw=1)
        ax.axvline(true_on, color="black", linestyle="--", label="GT onset")
        ax.axvline(true_off, color="black", linestyle="--", label="GT offset")
        ax.axvline(pred_on, color="red", linestyle="--", label="Pred onset")
        ax.axvline(pred_off, color="red", linestyle="--", label="Pred offset")

        ax.set_title(f"Signal {i+1}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        if i == 0:
            ax.legend(fontsize="xx-small")

    # Hide any empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_on_signal()
    print("All tests passed!")