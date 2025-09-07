# The idea of this class is to approximate the bursting bounds directly from the signal
# using a neural network. This is a test to see if we can do this without the dual threshold model.

#THIS FILE JUST WORKED. GOOD PREDICTIONS. 03:43 AM, SEPT 7, 2025.

from gru import Approximator, mse_loss
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.onnx import export

from sims_direct import (
    generate_training_data_approximator,
    fetch_real_data,
    fs,
)
from neurodsp.burst import detect_bursts_dual_threshold

from matplotlib import pyplot as plt


def run_on_signal():
    """
    Runs the Approximator model to approximate the parameters of the dual threshold model.


    """

    # apple silicon, nvidia, or cpu :)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    # TODO: change these sizes to be bigger
    train_size = 1000
    test_size = 100

    # ok so I switch val and test later. So the test data is the real data and val is simulated

    x_val, y_val, gt_y_val = generate_training_data_approximator(
        n_sims=test_size, mode="test"
    )
    x_train, y_train, gt_y_train = generate_training_data_approximator(
        n_sims=train_size, mode="train"
    )
    x_test, y_test, gt_y_test = fetch_real_data()

    # Use ground truth bounds as targets
    y_train = gt_y_train
    y_test = gt_y_test
    y_val = gt_y_val

    x_test, y_test, x_train, y_train = (
        np.array(x_test),
        np.array(y_test),
        np.array(x_train),
        np.array(y_train),
    )

    # length of signal
    input_shape = x_train.shape[1]
    print(f"Input shape: {input_shape}")

    # number of params to predict: currently 2: onset and offset
    # ideas: (onset, duration)
    output_shape = y_train.shape[1]

    hidden_dim = 120

    # Move data to the appropriate device
    x_train_tensor = torch.tensor(data=x_train, dtype=torch.float32).to(device)
    x_train_mean = torch.mean(x_train_tensor, dim=1, keepdim=True)
    x_train_std = torch.std(x_train_tensor, dim=1, keepdim=True)
    x_train_tensor = (x_train_tensor - x_train_mean) / (
        x_train_std + 1e-8
    )  # x_std >= 0 so add perturbation to prevent division by zero

    y_train_tensor = torch.tensor(data=y_train, dtype=torch.float32).to(device)
    y_train_mean = torch.mean(input=y_train_tensor, dim=1, keepdim=True)
    y_train_std = torch.std(input=y_train_tensor, dim=1, keepdim=True)
    y_train_tensor = (y_train_tensor - y_train_mean) / (y_train_std + 1e-8)

    x_val_tensor = torch.tensor(data=x_val, dtype=torch.float32).to(device)
    x_val_mean = torch.mean(x_val_tensor, dim=1, keepdim=True)
    x_val_std = torch.std(x_val_tensor, dim=1, keepdim=True)
    x_val_tensor = (x_val_tensor - x_val_mean) / (x_val_std + 1e-8)

    y_val_tensor = torch.tensor(data=y_val, dtype=torch.float32).to(device)
    y_val_mean = torch.mean(input=y_val_tensor, dim=1, keepdim=True)
    y_val_std = torch.std(input=y_val_tensor, dim=1, keepdim=True)
    y_val_tensor = (y_val_tensor - y_val_mean) / (y_val_std + 1e-8)

    # Ensuring the validation data is not used to train the model by turning off
    # gradient computation for validation data with torch.no_grad()
    # but still allowing it to be passed to the model during training for evaluation.

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Initialize and move model to device
    ml_approximator = Approximator(
        signal_dim=input_shape, param_dim=output_shape, hidden_dim=hidden_dim, lr=0.0001
    ).to(device)

    loss_train, loss_val = ml_approximator.train_model(
        train_loader, val_loader=val_loader, num_epochs=10
    )

    # Move the test data to the device
    # x_test should be signals
    x_test_tensor = torch.tensor(data=x_test, dtype=torch.float32).to(device)
    x_test_mean = torch.mean(x_test_tensor, dim=1, keepdim=True)
    x_test_std = torch.std(x_test_tensor, dim=1, keepdim=True)
    x_test_tensor = (x_test_tensor - x_test_mean) / (x_test_std + 1e-8)

    # y_test should be params
    y_test_ground_truth = torch.tensor(data=y_test, dtype=torch.float32).to("cpu")
    y_test_mean = torch.mean(input=y_test_ground_truth, dim=1, keepdim=True)
    y_test_std = torch.std(input=y_test_ground_truth, dim=1, keepdim=True)
    y_test_tensor = (y_test_ground_truth.to("cpu") - y_test_mean.to("cpu")).to("cpu") / (y_test_std.to("cpu") + 1e-8)

    # print(device)
    y_test_pred = ml_approximator.predict(x_test_tensor)

    # print(f"y_pred shape before reshape: {y_pred.shape}, expected: {y_test_tensor.shape}")

    if y_test_pred.shape != y_test_tensor.shape:
        y_test_pred = np.reshape(y_test_pred, y_test_tensor.shape)

    y_test_usable = y_test_tensor.cpu()

    mse = np.mean(
        np.square(
            y_test_pred.clone().cpu().numpy() - y_test_usable.clone().cpu().numpy()
        )
    )
    # print(y_test_pred.device, y_test_std.device, y_test_mean.device)
    # print(ytt_copy.device, y_test_std.device, y_test_mean.device)
    ytt_copy_denorm = (
        y_test_usable.clone().cpu().numpy() * y_test_std.clone().cpu().numpy()
        + y_test_mean.clone().cpu().numpy()
    )
    print(f"Mean Squared Error: {mse}")
    plt.figure()

    assert y_test_pred.shape == y_test_tensor.shape, "Prediction shape mismatch"

    # Step 1: De-normalize predictions
    y_test_pred_denorm = (y_test_pred.to(y_test_mean.device) * y_test_std.to(y_test_mean.device)).to(y_test_mean.device) + y_test_mean.to(y_test_mean.device)
    y_test_pred_denorm = y_test_pred_denorm.cpu().numpy()

    intervals_pred = []
    # Step 2: Compute predicted intervals
    for i in range(x_test.shape[0]):
        # Get predicted params
        pred_params = y_test_pred_denorm[i]
        
        # Extract theta values for detect_bursts
        burst_bounds = (
            pred_params[0],
            pred_params[1],
        )
        print(burst_bounds)

        # Run dual threshold detection on signal
        is_burst = detect_bursts_dual_threshold(x_test[i], fs, dual_thresh=[1,2], f_range=[5,10])

        # Find first burst interval
        bounds = [0, 0]
        found_onset = False
        found_offset = False
        for j in range(len(is_burst)):
            if not found_onset and is_burst[j] == 1:
                bounds[0] = j
                found_onset = True
            elif found_onset and not found_offset and is_burst[j] == 0:
                bounds[1] = j
                found_offset = True
                break

        intervals_pred.append(bounds)

        # Plot signal with predicted vs true intervals
        plt.subplot(test_size // 5, 5, i + 1)
        plt.plot(x_test[i], label="signal")
        plt.axvline(gt_y_test[i][0], color="black", linestyle="--", label="GT onset")
        plt.axvline(gt_y_test[i][1], color="black", linestyle="--", label="GT offset")
        plt.axvline(burst_bounds[0], color="red", linestyle="--", label="Pred onset")
        plt.axvline(burst_bounds[1], color="red", linestyle="--", label="Pred offset")
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.legend(fontsize="xx-small")
    plt.show()


if __name__ == "__main__":
    # test_approximator()
    run_on_signal()
    print("All tests passed!")
