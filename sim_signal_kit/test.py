from learning_system import Approximator, mse_dualthresh
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dualthresh_model import DualThreshModel
from sim_signals_dualthresh import generate_training_data_approximator, fs
from neurodsp.burst import detect_bursts_dual_threshold

from matplotlib import pyplot as plt


def test_approximator():
    # Check for MPS availability and set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cpu")

    train_size = 1000
    test_size = 100

    # Test case: approximate base of an exponential function
    x_train = np.zeros((train_size, 100))
    x_test = np.zeros((test_size, 100))
    y_train = np.zeros((train_size, 2))
    y_test = np.zeros((test_size, 2))

    for i in range(train_size):
        random_base = np.random.uniform(0.1, 2.0)
        random_scalar = np.random.uniform(-100, 100)
        x_train[i, :] = random_scalar * np.power(
            random_base, np.arange(x_train.shape[1])
        )
        y_train[i, 0] = random_base
        y_train[i, 1] = random_scalar

    for i in range(test_size):
        random_base = np.random.uniform(0.1, 2.0)
        random_scalar = np.random.uniform(-100, 100)
        x_test[i, :] = random_scalar * np.power(random_base, np.arange(x_test.shape[1]))
        x_test[i, :] += np.random.normal(0, 0.1, x_test.shape[1])

        # convert to float
        x_test[i, :] = x_test[i, :].astype(float)

        y_test[i, 0] = float(random_base)
        y_test[i, 1] = float(random_scalar)

    input_shape = x_train.shape[1]
    output_shape = y_train.shape[1]

    hidden_dim = 30

    # Move data to the appropriate device
    x_tensor = torch.tensor(data=x_train, dtype=torch.float32).to(device)
    x_mean = torch.mean(x_tensor, dim=0, keepdim=True)
    x_std = torch.std(x_tensor, dim=0, keepdim=True)
    x_tensor = (x_tensor - x_mean) / (
        x_std + 1e-8
    )  # Add small epsilon to prevent division by zero

    y_tensor = torch.tensor(data=y_train, dtype=torch.float32).to(device)
    y_mean = torch.mean(input=y_tensor, dim=0, keepdim=True)
    y_std = torch.std(input=y_tensor, dim=0, keepdim=True)
    y_tensor = (y_tensor - y_mean) / (y_std + 1e-8)

    train_dataset = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Initialize and move model to device
    approximator = Approximator(
        signal_dim=input_shape, param_dim=output_shape, hidden_dim=hidden_dim, lr=0.001
    ).to(device)

    approximator.train_model(train_loader, num_epochs=20)

    # Move the test data to the device
    x_test_tensor = torch.tensor(data=x_test, dtype=torch.float32).to(device)
    x_test_tensor = (x_test_tensor - x_mean) / (x_std + 1e-8)
    y_test_tensor = torch.tensor(data=y_test, dtype=torch.float32).to(device)
    y_test_tensor = (y_test_tensor - y_mean) / (y_std + 1e-8)

    y_pred = approximator.predict(x_test_tensor)

    print(
        f"y_pred shape before reshape: {y_pred.shape}, expected: {y_test_tensor.shape}"
    )

    if y_pred.shape != y_test_tensor.shape:
        y_pred = np.reshape(y_pred, y_test_tensor.shape)

    mse = np.mean(np.square(y_pred - y_test_tensor.cpu().numpy()))
    for i in range(y_pred.shape[0]):
        print(f"y_pred: {y_pred[i]}, y_test: {y_test_tensor[i]}")
    print(f"Mean Squared Error: {mse}")
    # assert np.allclose(
    #     y_pred, y_test_tensor.cpu().numpy(), atol=1e-1
    # ), f"Expected {y_test_tensor.cpu().numpy()}, but got {y_pred}"
    print("Test passed!")


def get_train_data():
    """
    Get Signal,param pairs for training.
    Requires optimizing for dualthresh, a lot of times.
    """


def get_test_data(test_size):
    """
    Get Signal,param pairs for testing.
    Ideally, the test_dataset is different from the train_dataset.
    Requires optimizing for dualthresh, a lot of times.
    """


# TODO before writing this segment: fill in get_data.
# it should work like: x_train, y_train=get_train_data(train_size)
# x_test, y_test=get_test_data(test_size)
def run_on_dualthresh():
    # Check for MPS availability and set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cpu")

    train_size = 1000
    test_size = 100

    x_test, y_test, gt_test = generate_training_data_approximator()
    x_train, y_train, gt_train = generate_training_data_approximator()

    x_test, y_test, x_train, y_train = (
        np.array(x_test),
        np.array(y_test),
        np.array(x_train),
        np.array(y_train),
    )
    input_shape = x_train.shape[1]
    output_shape = y_train.shape[1]

    hidden_dim = 30

    # Move data to the appropriate device
    x_train_tensor = torch.tensor(data=x_train, dtype=torch.float32).to(device)
    x_train_mean = torch.mean(x_train_tensor, dim=0, keepdim=True)
    x_train_std = torch.std(x_train_tensor, dim=0, keepdim=True)
    x_train_tensor = (x_train_tensor - x_train_mean) / (
        x_train_std + 1e-10
    )  # x_std >= 0 so add perturbation to prevent division by zero

    y_train_tensor = torch.tensor(data=y_train, dtype=torch.float32).to(device)
    y_train_mean = torch.mean(input=y_train_tensor, dim=0, keepdim=True)
    y_train_std = torch.std(input=y_train_tensor, dim=0, keepdim=True)
    y_train_tensor = (y_train_tensor - y_train_mean) / (y_train_std + 1e-8)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Initialize and move model to device
    approximator = Approximator(
        signal_dim=input_shape, param_dim=output_shape, hidden_dim=hidden_dim, lr=0.0001
    ).to(device)

    approximator.train_model(train_loader, num_epochs=100)

    # Move the test data to the device
    # x_test should be signals
    x_test_tensor = torch.tensor(data=x_test, dtype=torch.float32).to(device)
    x_test_mean = torch.mean(x_test_tensor, dim=0, keepdim=True)
    x_test_std = torch.std(x_test_tensor, dim=0, keepdim=True)
    x_test_tensor = (x_test_tensor - x_test_mean) / (x_test_std + 1e-8)

    # y_test should be params
    y_test_ground_truth = torch.tensor(data=y_test, dtype=torch.float32).to(device)
    y_test_mean = torch.mean(input=y_test_ground_truth, dim=0, keepdim=True)
    y_test_std = torch.std(input=y_test_ground_truth, dim=0, keepdim=True)
    y_test_tensor = (y_test_ground_truth - y_test_mean) / (y_test_std + 1e-8)

    # print(device)
    y_test_pred = approximator.predict(x_test_tensor)

    # print(f"y_pred shape before reshape: {y_pred.shape}, expected: {y_test_tensor.shape}")

    if y_test_pred.shape != y_test_tensor.shape:
        y_test_pred = np.reshape(y_test_pred, y_test_tensor.shape)

    ytt_copy = y_test_tensor.cpu()

    mse = np.mean(
        np.square(y_test_pred.clone().cpu().numpy() - ytt_copy.clone().cpu().numpy())
    )
    # print(y_test_pred.device, y_test_std.device, y_test_mean.device)
    y_pred_denorm = y_test_pred * y_test_std + y_test_mean
    # print(ytt_copy.device, y_test_std.device, y_test_mean.device)
    ytt_copy_denorm = (
        ytt_copy.clone().cpu().numpy() * y_test_std.clone().cpu().numpy()
        + y_test_mean.clone().cpu().numpy()
    )
    mse2 = mse_dualthresh(
        x_test_tensor.clone().cpu().numpy(),
        y_pred_denorm.clone().cpu().numpy(),
        y_true=y_test_ground_truth.clone().cpu().numpy(),
    )
    print(f"Mean Squared Error 2: {mse2}")
    plt.figure()

    yptccn = y_pred_denorm.clone().cpu().numpy()
    intervals_pred=[]
    for i in range(y_test_pred.shape[0]):
        print(f"result: {y_test_pred[i]}, expected: {ytt_copy.numpy()[i]}")
        assert np.all(y_test_pred[i][j]>=0 for j in range(4)), "All values should be positive"

        y_test_true_denormed_numpy = ytt_copy_denorm
        is_burst_pred = detect_bursts_dual_threshold(
            x_test[i],
            fs,
            (
                y_test_true_denormed_numpy[i][0],
                y_test_true_denormed_numpy[i][0]
                + y_test_true_denormed_numpy[i][1],
            ),
            (y_test_true_denormed_numpy[i][2], y_test_true_denormed_numpy[i][2] + y_test_true_denormed_numpy[i][3]),
        )

        for j in range(len(is_burst_pred)):
            bounds_list=[0,0]
            found_onset = False
            found_offset = False
            if not found_onset and is_burst_pred[j]==1:
                bounds_list[0] = j
                found_onset = True
            if found_onset and not found_offset and is_burst_pred[j]==0:
                bounds_list[1] = j
        interval_pred = np.array(bounds_list)
        intervals_pred.append(interval_pred)



        print(f"result: {intervals_pred[i]}, expected: {gt_test[i]}")
        plt.subplot(int(y_test_pred.shape[0] / 2), 2, i + 1)
        plt.plot(x_test[i], label="signal")
        plt.axvline(gt_test[i][0], color="C2")
        plt.axvline(gt_test[i][1], color="C2", label="true")
        plt.axvline(interval_pred[0], color="C3")
        plt.axvline(
            interval_pred[1], color="C3", label="predicted"
        )
        # plt.plot(y_test_tensor[i], label='true')
        plt.legend()
        # plot all predictions on signals, different subplots for different signals
    plt.show()

    intervals_pred = np.array(intervals_pred)

    print(f"Mean Squared Error: {mse}")
    assert np.allclose(
        gt_test,
        intervals_pred,
        atol=50,
    ), f"Expected {gt_test}, but got {intervals_pred}"
    print("Test passed!")


if __name__ == "__main__":
    # test_approximator()
    run_on_dualthresh()
    print("All tests passed!")
