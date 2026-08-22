import numpy as np


def softmax(Z):
    Z_shift = Z - np.max(Z, axis=-1, keepdims=True)
    E = np.exp(Z_shift)
    return E/np.sum(E, axis=-1, keepdims=True)


def softmax_cross_entropy(Z, y):
    N = Z.shape[0]
    softmax_Z = softmax(Z)
    p_correct = softmax_Z[np.arange(N), y]
    Loss = -np.mean(np.log(p_correct+1e-12))
    dZ = softmax_Z.copy()
    dZ[np.arange(N), y] -= 1
    dZ /= N
    return Loss, dZ


def accuracy(Z, y):
    return np.mean(np.argmax(Z, axis=1) == y)
