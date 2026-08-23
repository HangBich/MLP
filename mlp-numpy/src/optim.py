class SGD:
    def __init__(self, params_and_grads_fn, lr: float, momentum: float = 0.0):
        self.fn = params_and_grads_fn
        self.lr = lr

    def step(self):
        for p, g in self.fn():
            p -= self.lr*g
