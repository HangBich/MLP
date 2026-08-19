"""4 hàm kích hoạt bắt buộc so sánh: Sigmoid, Tanh, ReLU, Leaky ReLU.

Mỗi hàm cài theo cặp forward/backward. Quy ước backward:
    dZ = dA * f'(Z)
Truyền vào cả Z (pre-activation) và A (post-activation) để bạn tự chọn
cách tính đạo hàm rẻ hơn (vd sigmoid: f' = A*(1-A), khỏi tính lại exp).
"""
import numpy as np


def sigmoid(Z):
    """Ổn định số: tách nhánh Z >= 0 và Z < 0, tránh overflow exp(-Z)."""
    raise NotImplementedError


def sigmoid_backward(dA, Z, A):
    raise NotImplementedError


def tanh(Z):
    raise NotImplementedError


def tanh_backward(dA, Z, A):
    raise NotImplementedError


def relu(Z):
    raise NotImplementedError


def relu_backward(dA, Z, A):
    """Lưu ý điểm Z == 0: chọn 0 hay 1 đều được, nêu rõ lựa chọn trong báo cáo."""
    raise NotImplementedError


def leaky_relu(Z, slope: float = 0.01):
    raise NotImplementedError


def leaky_relu_backward(dA, Z, A, slope: float = 0.01):
    raise NotImplementedError


# Registry: config.activation -> (forward, backward)
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_backward),
    "tanh": (tanh, tanh_backward),
    "relu": (relu, relu_backward),
    "leaky_relu": (leaky_relu, leaky_relu_backward),
}


def get_activation(name: str, slope: float = 0.01):
    """Trả về (fwd, bwd) đã bind sẵn slope cho leaky_relu."""
    raise NotImplementedError
