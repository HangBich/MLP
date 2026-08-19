"""Tầng Linear và BatchNorm1D thuần NumPy.

Quy ước cache: mỗi tầng tự giữ những gì backward cần trong self.cache,
được ghi ở forward và đọc ở backward. Sau mỗi backward nên xóa cache
để tránh dùng nhầm dữ liệu của batch trước.
"""
import numpy as np


class Linear:
    """A = X @ W + b

    Shapes:  X (N, D_in) | W (D_in, D_out) | b (D_out,) | out (N, D_out)
    """

    def __init__(self, fan_in: int, fan_out: int, init_scheme: str, rng):
        self.W = None   # TODO: init_weights(...)
        self.b = None   # TODO: init_bias(...)
        self.dW = None
        self.db = None
        self.cache = None
        raise NotImplementedError

    def forward(self, X):
        raise NotImplementedError

    def backward(self, dOut):
        """Cho dOut (N, D_out), tính:
            self.dW (D_in, D_out), self.db (D_out,), trả về dX (N, D_in)

        Lưu ý chuẩn hóa theo batch: quyết định chia 1/N ở ĐÂY hay ở hàm loss,
        chỉ chọn MỘT chỗ. Chia hai lần là lỗi kinh điển khiến gradient check
        lệch đúng bằng hệ số N.
        """
        raise NotImplementedError

    def params_and_grads(self):
        """Trả về [(W, dW), (b, db)] cho optimizer."""
        raise NotImplementedError


class BatchNorm1D:
    """Phần 'Yêu cầu khác'. Cài CẢ forward VÀ backward bằng tay.

        mu    = mean(X, axis=0)
        var   = var(X, axis=0)
        Xhat  = (X - mu) / sqrt(var + eps)
        out   = gamma * Xhat + beta

    Backward: đạo hàm phải đi qua CẢ mu và var (chúng phụ thuộc X).
    Bỏ qua nhánh này là lỗi phổ biến nhất — gradient check sẽ bắt được.

    Chế độ train dùng thống kê batch; chế độ eval dùng running mean/var.
    """

    def __init__(self, dim: int, momentum: float = 0.9, eps: float = 1e-5):
        self.gamma = None       # TODO: khởi tạo 1
        self.beta = None        # TODO: khởi tạo 0
        self.running_mean = None
        self.running_var = None
        self.momentum = momentum
        self.eps = eps
        self.cache = None
        raise NotImplementedError

    def forward(self, X, training: bool = True):
        raise NotImplementedError

    def backward(self, dOut):
        """Trả về dX; đặt self.dgamma, self.dbeta."""
        raise NotImplementedError

    def params_and_grads(self):
        raise NotImplementedError
