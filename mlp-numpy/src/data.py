import gzip, os, urllib.request
import numpy as np


NUM_CLASSES = 10

BASE = "https://ossci-datasets.s3.amazonaws.com/mnist"   # mirror MNIST
FASHION = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"

FILES = {
    "train_X": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_X":  "t10k-images-idx3-ubyte.gz",
    "test_y":  "t10k-labels-idx1-ubyte.gz",
}

def _download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)

def _read_idx(path, is_image):
    with gzip.open(path, "rb") as f:
        data = f.read()
    if is_image:
        n = int.from_bytes(data[4:8], "big")
        return np.frombuffer(data, np.uint8, offset=16).reshape(n, 784).astype(np.float64)
    return np.frombuffer(data, np.uint8, offset=8).astype(np.int64)

def load_raw(dataset="fashion_mnist", data_dir="./data"):
    base = FASHION if dataset == "fashion_mnist" else BASE
    paths = {}
    for key, fname in FILES.items():
        p = os.path.join(data_dir, dataset, fname)
        _download(f"{base}/{fname}", p)
        paths[key] = p

    Xtr = _read_idx(paths["train_X"], True)
    ytr = _read_idx(paths["train_y"], False)
    Xte = _read_idx(paths["test_X"], True)
    yte = _read_idx(paths["test_y"], False)
    return (Xtr, ytr), (Xte, yte)


def preprocess(X_train, X_other_list, mode):
    if mode == "raw":
        return X_train, list(X_other_list), None, None
    if mode == "scale01":
        return X_train/255.0, [X/255.0 for X in X_other_list], None, None
    if mode == "standardize":
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0) + 1e-8
        return (X_train-mu)/sd, [(X-mu)/sd for X in X_other_list], mu, sd


def train_val_split(X, y, val_ratio: float, rng: np.random.Generator):
    n = len(X)
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def iterate_minibatches(X, y, batch_size: int, rng: np.random.Generator, shuffle: bool = True):
    n = len(X)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        batch = idx[start:start + batch_size]
        yield X[batch], y[batch]
