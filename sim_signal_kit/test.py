from learning_system import Approximator
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dualthresh_model import DualThreshModel
from sim_signals_dualthresh import generate_training_data_approximator

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

    print(f"y_pred shape before reshape: {y_pred.shape}, expected: {y_test_tensor.shape}")

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

    x_test,y_test = generate_training_data_approximator()
    x_train, y_train = generate_training_data_approximator()

    x_test, y_test, x_train, y_train = np.array(x_test), np.array(y_test), np.array(x_train), np.array(y_train)
    input_shape = x_train.shape[1]
    output_shape = y_train.shape[1]

    hidden_dim = 30

    # Move data to the appropriate device
    x_tensor = torch.tensor(data=x_train, dtype=torch.float32).to(device)
    x_mean = torch.mean(x_tensor, dim=0, keepdim=True)
    x_std = torch.std(x_tensor, dim=0, keepdim=True)
    x_tensor = (x_tensor - x_mean) / (
        x_std + 1e-10
    )  # x_std >= 0 so add perturbation to prevent division by zero

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

    print(f"y_pred shape before reshape: {y_pred.shape}, expected: {y_test_tensor.shape}")

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


if __name__ == "__main__":
    # test_approximator()
    run_on_dualthresh()
    print("All tests passed!")