"""5 cách khởi tạo bắt buộc so sánh.

    zeros         W = 0                      -> minh họa symmetry breaking
    normal_small  W ~ N(0, 0.01^2)
    normal_large  W ~ N(0, 1.0^2)            -> minh họa bão hòa (sigmoid/tanh)
    xavier        std = sqrt(1/fan_in)  hoặc sqrt(2/(fan_in+fan_out))
    he            std = sqrt(2/fan_in)       -> hợp với ReLU

Bias: khởi tạo 0 cho mọi trường hợp (nêu rõ trong báo cáo).
Mọi phép ngẫu nhiên PHẢI dùng rng truyền vào, không dùng np.random toàn cục.
"""
import numpy as np


def init_weights(fan_in: int, fan_out: int, scheme: str, rng: np.random.Generator):
    """Trả về W shape (fan_in, fan_out).

    Ghi rõ trong báo cáo bạn dùng biến thể Xavier nào (fan_in hay fan_avg).
    """
    raise NotImplementedError


def init_bias(fan_out: int):
    """Trả về b shape (fan_out,) — toàn 0."""
    raise NotImplementedError


INIT_SCHEMES = ["zeros", "normal_small", "normal_large", "xavier", "he"]
