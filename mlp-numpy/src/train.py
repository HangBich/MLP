"""Vòng lặp huấn luyện cho MỘT lượt chạy (một Config + một seed)."""
import numpy as np

from . import data as data_mod
from .config import Config
from .model import MLP
from .monitor import Monitor
from .optim import SGD
from .utils import save_history, set_seed, timer


def evaluate(model, X, y, batch_size: int = 512):
    """Trả về (loss, accuracy) ở chế độ eval. Chia batch để đỡ tốn RAM."""
    raise NotImplementedError


def train_one_run(cfg: Config) -> dict:
    """Chạy trọn một cấu hình. KHÔNG in bừa ra stdout — trả về dict để
    run_experiment.py ghi vào CSV.

    Các bước:
      1. rng = set_seed(cfg.seed)
      2. tải dữ liệu -> train/val split -> preprocess (fit trên train)
      3. dựng MLP, SGD, Monitor
      4. vòng lặp epoch:
           - iterate_minibatches -> forward -> loss -> backward -> step
           - cuối epoch: evaluate train/val, ghi vào history
           - nếu epoch % cfg.monitor_every == 0: thu thống kê activation/gradient
      5. đánh giá trên test
      6. save_history(...)

    Returns
    -------
    dict với các khóa khớp header của logs/runs.csv:
        train_acc, val_acc, test_acc, final_train_loss, final_val_loss,
        epochs_to_90pct_best, wall_time_sec, history_path
    """
    raise NotImplementedError


def epochs_to_threshold(val_acc_curve, frac: float = 0.9) -> int:
    """'Thời gian hội tụ' dùng để so sánh công bằng giữa các cấu hình:
    số epoch đầu tiên đạt >= frac * (val_acc tốt nhất của chính lượt chạy đó).
    Trả về -1 nếu không bao giờ đạt.
    """
    raise NotImplementedError
