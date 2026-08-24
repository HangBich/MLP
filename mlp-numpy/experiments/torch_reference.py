"""Đối chiếu bản NumPy với PyTorch để kiểm chứng tính đúng đắn."""
import numpy as np
import torch
import torch.nn as nn

from src.config import Config
from src.losses import softmax_cross_entropy
from src.model import MLP
from src.utils import set_seed

rng = set_seed(0)
cfg = Config(activation="relu", init="he", hidden_sizes=[256, 128, 64])
mynet = MLP(784, cfg.hidden_sizes, 10, cfg, rng)

# 1. Dựng mạng PyTorch cùng kiến trúc
torch_net = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 64),  nn.ReLU(),
    nn.Linear(64, 10),
)
torch_net = torch_net.double()


# 2. Copy trọng số — TRANSPOSE
my_linears = [l for l in mynet.layers if hasattr(l, "W")]
torch_linears = [m for m in torch_net if isinstance(m, nn.Linear)]

with torch.no_grad():
    for ml, tl in zip(my_linears, torch_linears):
        tl.weight.copy_(torch.tensor(ml.W.T))   # (D_in,D_out) -> (D_out,D_in)
        tl.bias.copy_(torch.tensor(ml.b))

# 3. Cùng một batch qua cả hai
X = rng.standard_normal((16, 784))
y = rng.integers(0, 10, size=16)

logits_np = mynet.forward(X)
Xt = torch.tensor(X, dtype=torch.float64)
logits_t = torch_net(Xt)

print("Lệch logits :", np.abs(logits_np - logits_t.detach().numpy()).max())

# 4. So gradient
loss_np, dZ = softmax_cross_entropy(logits_np, y)
mynet.backward(dZ)

loss_t = nn.functional.cross_entropy(logits_t, torch.tensor(y))
loss_t.backward()

print("Lệch loss   :", abs(loss_np - loss_t.item()))
for i, (ml, tl) in enumerate(zip(my_linears, torch_linears)):
    dw = np.abs(ml.dW - tl.weight.grad.numpy().T).max()
    db = np.abs(ml.db - tl.bias.grad.numpy()).max()
    print(f"Tầng {i}: lệch dW = {dw:.2e}   lệch db = {db:.2e}")

# 5. So loss qua vài bước SGD
from src.optim import SGD

print("\n--- 5 bước SGD, lr = 0.1 ---")
opt_np = SGD(mynet.params_and_grads, lr=0.1)
opt_t = torch.optim.SGD(torch_net.parameters(), lr=0.1)
yt = torch.tensor(y)

for step in range(5):
    loss_np, dZ = softmax_cross_entropy(mynet.forward(X), y)
    mynet.backward(dZ)
    opt_np.step()

    opt_t.zero_grad()
    loss_t = nn.functional.cross_entropy(torch_net(Xt), yt)
    loss_t.backward()
    opt_t.step()

    print(f"bước {step}:  numpy {loss_np:.12f}   torch {loss_t.item():.12f}   "
          f"lệch {abs(loss_np - loss_t.item()):.2e}")
