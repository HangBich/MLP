"""Minibatch SGD (bắt buộc). Momentum là tùy chọn nếu nhóm muốn mở rộng."""


class SGD:
    def __init__(self, params_and_grads_fn, lr: float, momentum: float = 0.0):
        """params_and_grads_fn: callable trả về list các cặp (param, grad),
        param được cập nhật IN-PLACE (dùng p -= ... chứ không p = p - ...)."""
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        """Không bắt buộc nếu backward luôn ghi đè grad, nhưng nên có."""
        raise NotImplementedError
