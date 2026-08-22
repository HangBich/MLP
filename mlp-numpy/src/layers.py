import numpy as np
from .activations import get_activation

class Activation:
    def __init__(self, name, slope=0.01):
        self.fwd, self.bwd = get_activation(name, slope)
        self.cache = None

    def forward(self, X):
        A = self.fwd(X)
        self.cache = (X, A)
        return A

    def backward(self, dOut):
        Z, A = self.cache
        return self.bwd(dOut, Z, A)

    def params_and_grads(self):
        return []


class Linear:
    """A = X @ W + b

    Shapes:  X (N, D_in) | W (D_in, D_out) | b (D_out,) | out (N, D_out)
    """

    def __init__(self, fan_in: int, fan_out: int, init_scheme: str, rng):
        std = np.sqrt(2.0 / fan_in)
        self.W =  rng.normal(0, std, size=(fan_in, fan_out))
        self.b = np.zeros(fan_out)
        self.dW = None
        self.db = None
        self.cache = None

    def forward(self, X):
        self.cache = X
        return X @ self.W + self.b

    def backward(self, dOut):
        X = self.cache 
        self.dW = X.T @ dOut 
        self.db = np.sum(dOut, axis=0)
        dX = dOut @ self.W.T
        return dX 

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class BatchNorm1D:
    """Phần 'Yêu cầu khác'. Cài CẢ forward VÀ backward bằng tay.

        mu    = mean(X, axis=0)
        var   = var(X, axis=0)
        Xhat  = (X - mu) / sqrt(var + eps)
        out   = gamma * Xhat + beta

    Backward: đạo hàm phải đi qua CẢ mu và var (chúng phụ thuộc X).
    Bỏ qua nhánh này là lỗi phổ biến nhất — gradient check sẽ bắt được.

    Chế độ train dùng thống kê batch; chế độ eval dùng running mean/var.
    """

    def __init__(self, dim: int, momentum: float = 0.9, eps: float = 1e-5):
        self.gamma = None       # TODO: khởi tạo 1
        self.beta = None        # TODO: khởi tạo 0
        self.running_mean = None
        self.running_var = None
        self.momentum = momentum
        self.eps = eps
        self.cache = None
        raise NotImplementedError

    def forward(self, X, training: bool = True):
        raise NotImplementedError

    def backward(self, dOut):
        """Trả về dX; đặt self.dgamma, self.dbeta."""
        raise NotImplementedError

    def params_and_grads(self):
        raise NotImplementedError
