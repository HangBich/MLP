import json
import numpy as np

from src.config import Config
from src.data import iterate_minibatches, load_raw, preprocess, train_val_split
from src.layers import Activation, Linear
from src.losses import softmax_cross_entropy
from src.model import MLP
from src.optim import SGD
from src.utils import set_seed

records = []

for act in ["sigmoid", "tanh", "relu", "leaky_relu"]:
    cfg = Config(activation=act, init="xavier", epochs=3, seed=0)
    rng = set_seed(cfg.seed)

    (Xtr, ytr), (Xte, yte) = load_raw(cfg.dataset)
    Xtr, ytr, Xval, yval = train_val_split(Xtr, ytr, cfg.val_ratio, rng)
    Xtr, (Xval, Xte) = preprocess(Xtr, [Xval, Xte], cfg.preprocess)

    model = MLP(Xtr.shape[1], cfg.hidden_sizes, 10, cfg, rng)
    opt = SGD(model.params_and_grads, cfg.lr)

    for ep in range(cfg.epochs):
        for Xb, yb in iterate_minibatches(Xtr, ytr, cfg.batch_size, rng):
            loss, dZ = softmax_cross_entropy(model.forward(Xb), yb)
            model.backward(dZ)
            opt.step()
        # sau mỗi epoch, cache và dW còn giữ giá trị của batch cuối
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Activation):
                Z, A = layer.cache
                counts, edges = np.histogram(A, bins=50)
                records.append({
                    "act": act, "epoch": ep, "layer": i, "kind": "activation",
                    "mean": float(A.mean()), "std": float(A.std()),
                    "dead_frac": float(np.mean(np.abs(A) < 1e-8)),
                    "sat_frac": float(np.mean(np.abs(A) > 0.99)),
                    "hist": counts.tolist(), "edges": edges.tolist(),
                })
            elif isinstance(layer, Linear):
                records.append({
                    "act": act, "epoch": ep, "layer": i, "kind": "gradient",
                    "grad_norm": float(np.linalg.norm(layer.dW)),
                    "grad_rms": float(np.linalg.norm(layer.dW) / np.sqrt(layer.dW.size)),
                })

with open("results/activation_gradient_stats.json", "w") as f:
    json.dump(records, f)
print(f"Đã lưu {len(records)} bản ghi")