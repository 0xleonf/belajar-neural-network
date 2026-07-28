import torch
from torch import Tensor

# input
X = torch.tensor([1.0, 2.0], requires_grad=True)

# target
y = 1.5

# bobot awal
bobot = torch.tensor([0.2, -0.3, 0.4, 0.1, 0.5, -0.4], requires_grad=True)
bias = torch.tensor([0.1, -0.2, 0.0], requires_grad=True)


# fungsi linear
def linear(x1: float, x2: float, bobot1: float, bobot2: float, bias: float) -> float:
    return (bobot1 * x1) + (bobot2 * x2) + bias


# fungsi aktivasi
def ReLU(z: float) -> float:
    return torch.where(z < 0, 0, z)


def forward_pass(X: Tensor, bobot: Tensor, bias: Tensor) -> float:
    z1 = linear(X[0], X[1], bobot[0], bobot[1], bias[0])
    z2 = linear(X[0], X[1], bobot[2], bobot[3], bias[1])

    relu1 = ReLU(z1)
    relu2 = ReLU(z2)

    # output layer
    prediksi = (bobot[4] * relu1) + (bobot[5] * relu2) + bias[2]

    return prediksi, relu1, relu2


def loss_function(prediksi: float, target: float) -> float:
    return 1/2 * ((prediksi - target)**2)


def backpropagation(prediksi: float) -> float:
    error = loss_function(prediksi, y)
    error.backward()

    dw11, dw12, dw21, dw22, dwo1, dwo2 = bobot.grad
    db1, db2, dbo = bias.grad

    return dw11, dw12, dw21, dw22, dwo1, dwo2, db1, db2, dbo


def gradient_descent(X: Tensor, bobot: Tensor, bias: Tensor, lr: float) -> Tensor:
    target = 1.5
    for i in range(1000):
        hasil, relu1, relu2 = forward_pass(X, bobot, bias)

        print(f"iterasi: {
            i+1}, \t error: {loss_function(hasil, target):.12f}, \t prediksi: {hasil:.12f}")

        dw11, dw12, dw21, dw22, dwo1, dwo2, db1, db2, dbo = backpropagation(
            hasil)

        with torch.no_grad():
            bobot[0] -= lr * dw11
            bobot[1] -= lr * dw12
            bobot[2] -= lr * dw21
            bobot[3] -= lr * dw22
            bobot[4] -= lr * dwo1
            bobot[5] -= lr * dwo2
            bias[0] -= lr * db1
            bias[1] -= lr * db2
            bias[2] -= lr * dbo

            bobot.grad.zero_()
            bias.grad.zero_()

    return bobot, bias


gradient_descent(X, bobot, bias, 0.01)

# lin = linear(X[0], X[1], bobot[0], bobot[1], bias[0])
# fungsi = ReLU(lin)
# lin.retain_grad()
#
# fungsi.backward()
# print(f"turunan ReLU: {lin.grad}")
# print(f"turunan fungsi linear terhadap x1, x2: {X.grad}")
# print(f"turunan terhadap b1:
