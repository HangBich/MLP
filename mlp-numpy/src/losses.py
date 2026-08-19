"""Softmax + Cross-Entropy."""
import numpy as np


def softmax(Z):
    """Ổn định số: trừ max theo từng hàng trước khi exp.
    Bỏ bước này sẽ tràn số khi dùng khởi tạo normal_large.
    Z (N, C) -> (N, C)
    """
    raise NotImplementedError


def softmax_cross_entropy(Z, y):
    """Gộp softmax và cross-entropy làm một để ổn định số và để gradient gọn.

    Z : (N, C) logits (đầu ra tầng cuối, CHƯA qua softmax)
    y : (N,) nhãn nguyên

    Returns
    -------
    loss  : float, trung bình trên batch
    dZ    : (N, C), chính là (softmax(Z) - onehot(y)) / N

    Nhớ eps trong log để tránh log(0).
    """
    raise NotImplementedError


def accuracy(Z, y):
    raise NotImplementedError
