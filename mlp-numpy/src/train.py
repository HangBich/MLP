import numpy as np

from . import data as data_mod
from .config import Config
from .model import MLP
from .optim import SGD
from .utils import save_history, set_seed, timer
from .losses import softmax_cross_entropy


def evaluate(model, X, y, batch_size=512):
    n = len(X)
    total_loss, total_correct = 0.0, 0
    for start in range(0, n, batch_size):
        Xb, yb = X[start:start+batch_size], y[start:start+batch_size]
        logits = model.forward(Xb, training=False)
        loss, _ = softmax_cross_entropy(logits, yb)
        total_loss += loss * len(Xb)
        total_correct += np.sum(np.argmax(logits, axis=1) == yb)
    return total_loss / n, total_correct / n


def train_one_run(cfg: Config) -> dict:
    rng = set_seed(cfg.seed)

    (Xtr, ytr), (Xte, yte) = data_mod.load_raw(cfg.dataset)
    Xtr, ytr, Xval, yval = data_mod.train_val_split(Xtr, ytr, cfg.val_ratio, rng)
    Xtr, (Xval, Xte), mu, sd = data_mod.preprocess(Xtr, [Xval, Xte], cfg.preprocess)

    model = MLP(Xtr.shape[1], cfg.hidden_sizes, 10, cfg, rng)
    opt = SGD(model.params_and_grads, cfg.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    with timer() as t:
        for epoch in range(cfg.epochs):
            for Xb, yb in data_mod.iterate_minibatches(Xtr, ytr, cfg.batch_size, rng):
                loss, dZ = softmax_cross_entropy(model.forward(Xb), yb)
                model.backward(dZ)
                opt.step()

            tr1, tra = evaluate(model, Xtr, ytr)
            v1, va = evaluate(model, Xval, yval)
            history["train_loss"].append(tr1); history["train_acc"].append(tra)
            history["val_loss"].append(v1); history["val_acc"].append(va)

    tel, tea = evaluate(model, Xte, yte)
    path = save_history(cfg.run_id(), {"config": cfg.to_dict(), **history})

    params = {}
    for i, l in enumerate(model.layers):
        if hasattr(l, "W"):
            params[f"W{i}"] = l.W
            params[f"b{i}"] = l.b
    np.savez(f"results/model_{cfg.run_id()}.npz", mu=mu, sd=sd, **params)
    return {
        "train_acc": history["train_acc"][-1], 
        "val_acc": history["val_acc"][-1],
        "test_acc": tea, 
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "epochs_to_90pct_best": epochs_to_threshold(history["val_acc"]),
        "wall_time_sec": round(t["elapsed"], 2),
        "history_path": path,
    }


def epochs_to_threshold(val_acc_curve, frac: float = 0.9) -> int:
    best = max(val_acc_curve)
    for i, a in enumerate(val_acc_curve):
        if a>= frac * best:
            return i+1 
    return -1
