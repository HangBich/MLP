# # test_train.py  (ở mlp-numpy/, không phải src/)
# from src.config import Config
# from src.train import train_one_run

# cfg = Config(epochs=2, hidden_sizes=[64,32,16], activation="relu", init="he")
# print(train_one_run(cfg))

import numpy as np
from src.initializers import init_weights, INIT_SCHEMES
rng = np.random.default_rng(0)
for s in INIT_SCHEMES:
    print(s, init_weights(784, 256, s, rng).std().round(4))
