import torch
from torch import Tensor
import numpy as np
import pandas as pd


def init_params() -> Tensor:
    input_features = 784
    hidden_1 = 128
    hidden_2 = 64
    output_class = 10

    W1 = torch.rand(hidden_1, input_features,
                    dtype=torch.float32) - 0.5
    W1.requires_grad_(True)
    b1 = torch.rand(hidden_1, 1,  dtype=torch.float32) - 0.5
    b1.requires_grad_(True)

    W2 = torch.rand(hidden_2, hidden_1,  dtype=torch.float32) - 0.5
    W2.requires_grad_(True)
    b2 = torch.rand(hidden_2, 1,  dtype=torch.float32) - 0.5
    b2.requires_grad_(True)

    W3 = torch.rand(output_class, hidden_2, dtype=torch.float32) - 0.5
    W3.requires_grad_(True)
    b3 = torch.rand(output_class, 1,  dtype=torch.float32) - 0.5
    b3.requires_grad_(True)
    return W1, b1, W2, b2, W3, b3


def leaky_ReLU(Z: Tensor) -> Tensor:
    return torch.where(Z <= 0, (0.01 * Z), Z)


def softmax(Z: Tensor) -> Tensor:
    A = torch.exp(Z) / sum(torch.exp(Z))
    return A


def forward_pass(W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor, W3: Tensor, b3: Tensor, X: Tensor) -> Tensor:
    Z1 = torch.matmul(W1, X) + b1
    A1 = leaky_ReLU(Z1)
    Z2 = torch.matmul(W2, A1) + b2
    A2 = leaky_ReLU(Z2)
    Z3 = torch.matmul(W3, A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3


def one_hot_encode(Y: Tensor) -> Tensor:
    num_sample = Y.shape[0]
    one_hot = torch.zeros(
        (num_sample, int(Y.max().item() + 1)), dtype=torch.float32)
    one_hot[torch.arange(num_sample), Y.long()] = 1
    one_hot = one_hot.T
    return one_hot


def compute_loss(A: Tensor, Y: Tensor) -> float:
    m = Y.shape[0]
    one_hot_Y = one_hot_encode(Y)
    loss = -torch.sum(one_hot_Y * torch.log(A + 1e-8))/m
    return loss


def backpropagation(A3: Tensor, X: Tensor, Y: Tensor) -> Tensor:
    error = compute_loss(A3, Y)
    error.backward()
    # dW3 = W3.grad
    # db3 = b3.grad
    # dW2 = W2.grad
    # db2 = b2.grad
    # dW1 = W1.grad
    # db1 = b1.grad
    # return dW3, db3, dW2, db2, dW1, db1


def update_params(W1: Tensor, b1: Tensor, W2: Tensor, b2: Tensor, W3: Tensor, b3: Tensor, dW1: Tensor, db1: Tensor, dW2: Tensor, db2: Tensor, dW3: Tensor, db3: Tensor, alpha: torch.float32) -> Tensor:
    with torch.no_grad():
        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2
        W3 -= alpha * dW3
        b3 -= alpha * db3

        W1.grad.zero_()
        b1.grad.zero_()
        W2.grad.zero_()
        b2.grad.zero_()
        W3.grad.zero_()
        b3.grad.zero_()

    return W1, b1, W2, b2, W3, b3


def get_predictions(A3: Tensor) -> Tensor:
    return torch.argmax(A3, 0)


def get_accuracy(predictions: Tensor, Y: Tensor) -> float:
    print(predictions, Y)
    return torch.sum(predictions == Y) / Y.shape[0]


def gradient_descent(X: Tensor, Y: Tensor, alpha: torch.float32, iter: int) -> Tensor:
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iter):
        Z1, A1, Z2, A2, Z3, A3 = forward_pass(W1, b1, W2, b2, W3, b3, X)
        backpropagation(
            A3, X, Y)
        W1, b1, W2, b2, W3, b3, =  update_params(
            W1, b1, W2, b2, W3, b3, W1.grad, b1.grad, W2.grad, b2.grad, W3.grad, b3.grad, alpha)

        if i % 10 == 0:
            print(f"iterasi: {i}")
            predicts = get_predictions(A3)
            # error = compute_loss(A3, Y)
            # print(f"error: {error}")
            akurasi = get_accuracy(predicts, Y)
            print(f"akurasi: {akurasi}")

    return W1, b1, W2, b2, W3, b3


if __name__ == "__main__":
    data = pd.read_csv("../dataset/train.csv")
    m, n = data.shape
    data = np.array(data)
    np.random.shuffle(data)
    data = torch.tensor(data, dtype=torch.float32)

    data_validation = data[0:1000].T
    Y_val = data_validation[0]
    X_val = data_validation[1:n]
    X_val = X_val / 255.

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.

    W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.1, 1000)

    # Save model
    checkpoint = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3
    }
    torch.save(checkpoint, "ann_params.pt")
    print("Parameter sucessfully saved!")
