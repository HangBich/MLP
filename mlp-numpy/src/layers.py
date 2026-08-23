import numpy as np
from .activations import get_activation
from .initializers import init_weights, init_bias

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
    def __init__(self, fan_in: int, fan_out: int, init_scheme: str, rng):
        self.W =  init_weights(fan_in, fan_out, init_scheme, rng)
        self.b = init_bias(fan_out)
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
