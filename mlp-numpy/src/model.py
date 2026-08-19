"""MLP thuần NumPy: >= 3 tầng ẩn.

Kiến trúc mặc định (hidden_sizes = [256, 128, 64]):
    784 -> [Linear -> (BN) -> Act] x3 -> Linear -> 10 logits
Tầng cuối KHÔNG có activation (logits đi thẳng vào softmax_cross_entropy).
"""
import numpy as np


class MLP:
    def __init__(self, input_dim: int, hidden_sizes, num_classes: int, cfg, rng):
        """Dựng danh sách tầng theo cfg.activation / cfg.init / cfg.use_bn.

        Nên giữ self.layers là list các object có .forward/.backward để
        forward và backward chỉ là vòng lặp xuôi/ngược — dễ debug hơn nhiều
        so với viết tay từng tầng.
        """
        self.layers = []
        self.cfg = cfg
        raise NotImplementedError

    def forward(self, X, training: bool = True):
        """Trả về logits (N, C). Nếu self._monitor bật thì ghi lại
        pre-activation Z và post-activation A của từng tầng (xem monitor.py)."""
        raise NotImplementedError

    def backward(self, dZ):
        """Lan truyền ngược qua các tầng theo thứ tự đảo.
        Trả về dX (thường không dùng, nhưng cần cho gradient check trên input)."""
        raise NotImplementedError

    def params_and_grads(self):
        """Gom cặp (param, grad) của mọi tầng — dùng cho optimizer và gradcheck."""
        raise NotImplementedError

    def predict(self, X):
        """argmax logits ở chế độ eval (training=False, quan trọng khi có BN)."""
        raise NotImplementedError
