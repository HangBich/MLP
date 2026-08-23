import numpy as np
from src.config import Config
from src.model import MLP
from src.gradcheck import check_layer_gradients, format_report

cfg = Config(activation="tanh", init="he")
m = MLP(20, [5, 4, 3], 10, cfg, np.random.default_rng(0))
X = np.random.default_rng(1).standard_normal((10, 20))
y = np.random.default_rng(2).integers(0, 10, size=10)
print(format_report(check_layer_gradients(m, X, y)))