"""5 cách khởi tạo bắt buộc so sánh.

    zeros         W = 0                      -> minh họa symmetry breaking
    normal_small  W ~ N(0, 0.01^2)
    normal_large  W ~ N(0, 1.0^2)            -> minh họa bão hòa (sigmoid/tanh)
    xavier        std = sqrt(1/fan_in)  hoặc sqrt(2/(fan_in+fan_out))
    he            std = sqrt(2/fan_in)       -> hợp với ReLU

Bias: khởi tạo 0 cho mọi trường hợp.
Mọi phép ngẫu nhiên PHẢI dùng rng truyền vào, không dùng np.random toàn cục.
"""
import numpy as np


def init_weights(fan_in: int, fan_out: int, scheme: str, rng: np.random.Generator):
    if scheme == "zeros": std = 0.0
    elif scheme == "normal_small": std = 0.01
    elif scheme == "normal_large": std = 1.0
    elif scheme == "xavier": std = np.sqrt(1.0/fan_in)
    elif scheme == "he": std = np.sqrt(2.0/fan_in)
    else: raise ValueError(scheme)
    if std == 0.0: 
        return np.zeros((fan_in, fan_out))
    return rng.normal(0, std, size=(fan_in, fan_out))


def init_bias(fan_out: int):
    return np.zeros(fan_out)

INIT_SCHEMES = ["zeros", "normal_small", "normal_large", "xavier", "he"]
