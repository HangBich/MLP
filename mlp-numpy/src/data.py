"""Tải Fashion-MNIST và 3 chế độ tiền xử lý bắt buộc.

Quy ước shape dùng xuyên suốt project:
    X : (N, 784)  float64
    y : (N,)      int   nhãn 0..9
"""
import numpy as np

NUM_CLASSES = 10


def load_raw(dataset: str = "fashion_mnist", data_dir: str = "./data"):
    """Tải dữ liệu thô, KHÔNG tiền xử lý. Giá trị pixel ở [0, 255].

    Gợi ý: dùng torchvision.datasets.FashionMNIST(download=True) rồi
    .data.numpy().reshape(N, -1).astype(np.float64) và .targets.numpy().

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
    """
    raise NotImplementedError


def preprocess(X_train, X_other_list, mode: str):
    """Ba chế độ bắt buộc so sánh.

    mode = "raw"          -> giữ nguyên [0, 255]
    mode = "scale01"      -> chia 255
    mode = "standardize"  -> zero-mean + unit-variance

    !!! QUAN TRỌNG: mean/std phải tính TRÊN TẬP TRAIN rồi áp cho val/test.
    Tính trên toàn bộ dữ liệu là rò rỉ thông tin -> mất điểm phần thí nghiệm.
    Nhớ cộng eps vào std để tránh chia 0 ở các pixel viền luôn bằng 0.

    Returns
    -------
    X_train_p, [X_other_p, ...]
    """
    raise NotImplementedError


def train_val_split(X, y, val_ratio: float, rng: np.random.Generator):
    """Tách train/val bằng rng đã seed (không dùng np.random toàn cục)."""
    raise NotImplementedError


def one_hot(y, num_classes: int = NUM_CLASSES):
    """(N,) -> (N, num_classes)"""
    raise NotImplementedError


def iterate_minibatches(X, y, batch_size: int, rng: np.random.Generator, shuffle: bool = True):
    """Generator sinh (X_batch, y_batch).

    Thứ tự shuffle PHẢI lấy từ rng để tái lập được. Xử lý cả batch cuối lẻ.
    """
    raise NotImplementedError
