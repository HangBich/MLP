import numpy as np
from .layers import Activation, BatchNorm1D, Linear

class MLP:
    def __init__(self, input_dim: int, hidden_sizes, num_classes: int, cfg, rng):
        self.layers = []
        self.cfg = cfg

        dims = [input_dim] + list(hidden_sizes) + [num_classes]

        for i in range(len(dims)-1):
            self.layers.append(Linear(dims[i], dims[i+1], cfg.init, rng))
            if i < len(dims) - 2: 
                if cfg.use_bn:
                    self.layers.append(BatchNorm1D(dims[i+1]))
                self.layers.append(Activation(cfg.activation, cfg.leaky_slope))

    def forward(self, X, training: bool = True):
        for layer in self.layers:
            if isinstance(layer, BatchNorm1D):
                X = layer.forward(X, training=training)
            else:
                X = layer.forward(X)
        return X

    def backward(self, dZ):
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)
        return dZ

    def params_and_grads(self):
        out = []
        for layer in self.layers:
            out += layer.params_and_grads()
        return out 

    def predict(self, X):
        return np.argmax(self.forward(X, training=False), axis=1)
