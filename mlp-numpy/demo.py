import glob
import matplotlib.pyplot as plt
import numpy as np

from src.activations import relu
from src.data import load_raw

CLASSES = ["Áo phông", "Quần dài", "Áo chui đầu", "Váy", "Áo khoác",
           "Dép", "Áo sơ mi", "Giày thể thao", "Túi xách", "Bốt cổ ngắn"]

# 1. Load trọng số
d = np.load(sorted(glob.glob("results/model_demo*.npz"))[-1])
Ws = [d[f"W{i}"] for i in sorted(int(k[1:]) for k in d.files if k.startswith("W"))]
bs = [d[f"b{i}"] for i in sorted(int(k[1:]) for k in d.files if k.startswith("b"))]
mu, sd = d["mu"], d["sd"]

# 2. Forward thủ công (ReLU giữa các tầng, tầng cuối không activation)
def predict(X):
    A = (X - mu) / sd
    for i, (W, b) in enumerate(zip(Ws, bs)):
        A = A @ W + b
        if i < len(Ws) - 1:
            A = relu(A)
    return A

# 3. Lấy 8 ảnh test ngẫu nhiên
(_, _), (Xte, yte) = load_raw()
# Đổi seed slide 42, 
rng = np.random.default_rng(30)
idx = rng.choice(len(Xte), 8, replace=False)

logits = predict(Xte[idx])
pred = np.argmax(logits, axis=1)
prob = np.exp(logits - logits.max(axis=1, keepdims=True))
prob = prob / prob.sum(axis=1, keepdims=True)

# 4. Vẽ
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for k, ax in enumerate(axes.flat):
    i = idx[k]
    ok = pred[k] == yte[i]
    ax.imshow(Xte[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"Thật: {CLASSES[yte[i]]}\nĐoán: {CLASSES[pred[k]]} ({prob[k, pred[k]]:.0%})",
                 fontsize=10, color="green" if ok else "red")
    ax.axis("off")
fig.suptitle("Dự đoán trên 8 ảnh test ngẫu nhiên")
fig.tight_layout()
fig.savefig("results/figures/fig8_demo_predictions.png", dpi=150)
plt.show()