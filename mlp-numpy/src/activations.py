import numpy as np


def sigmoid(Z):
    out = np.empty_like(Z, dtype=np.float64)
    pos = Z >= 0 
    out[pos] = 1.0/(1.0+np.exp(-Z[pos]))
    ez = np.exp(Z[~pos])
    out[~pos] = ez/(1.0+ez)
    return out


def sigmoid_backward(dA, Z, A):
    return dA * A * (1-A)


def tanh(Z):
    return np.tanh(Z)


def tanh_backward(dA, Z, A):
    return dA * (1-A**2)


def relu(Z):
    return np.maximum(Z, 0)


def relu_backward(dA, Z, A):
    return dA*(Z>0)


def leaky_relu(Z, slope: float = 0.01):
    return np.where(Z>0, Z, slope * Z)

def leaky_relu_backward(dA, Z, A, slope: float = 0.01):
    return dA * np.where(Z > 0, 1.0, slope)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_backward),
    "tanh": (tanh, tanh_backward),
    "relu": (relu, relu_backward),
    "leaky_relu": (leaky_relu, leaky_relu_backward),
}


def get_activation(name: str, slope: float = 0.01):
    """Trả về (fwd, bwd) đã bind sẵn slope cho leaky_relu."""
    fwd, bwd = ACTIVATIONS[name]
    if name == "leaky_relu":
        return (lambda Z: fwd(Z, slope),
                lambda dA, Z, A: bwd(dA, Z, A, slope))
    return fwd, bwd

if __name__ == "__main__":
    Z = np.array([[-1000., -1., 0., 1., 1000.]])
    for name in ACTIVATIONS:
        f, b = get_activation(name)
        A = f(Z)
        print(name, A.round(3), b(np.ones_like(Z), Z, A).round(3))
