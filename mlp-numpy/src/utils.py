"""Tiện ích chung: seed, đo thời gian, ghi log CSV, lưu history."""
import csv, json, os, subprocess, time
from contextlib import contextmanager
from datetime import datetime

import numpy as np

LOG_CSV = os.path.join(os.path.dirname(__file__), "..", "logs", "runs.csv")
HIST_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "histories")


def set_seed(seed: int) -> np.random.Generator:
    """Cố định seed. TRẢ VỀ generator — hãy truyền generator này vào MỌI chỗ
    cần ngẫu nhiên (khởi tạo trọng số, shuffle minibatch, dropout) thay vì gọi
    np.random.* toàn cục, nếu không sẽ không tái lập được."""
    np.random.seed(seed)
    return np.random.default_rng(seed)


@contextmanager
def timer():
    t0 = time.perf_counter()
    box = {}
    yield box
    box["elapsed"] = time.perf_counter() - t0


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def append_run(row: dict) -> None:
    """Ghi 1 dòng vào logs/runs.csv. Cột nào thiếu thì để trống.
    Đây là 'Nhật ký thí nghiệm' bắt buộc nộp — thiếu là -10 điểm."""
    path = os.path.abspath(LOG_CSV)
    with open(path, "r", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=header).writerow(
            {k: row.get(k, "") for k in header}
        )


def save_history(run_id: str, history: dict) -> str:
    """Lưu đường cong loss/acc theo epoch + thống kê activation/gradient."""
    os.makedirs(os.path.abspath(HIST_DIR), exist_ok=True)
    path = os.path.join(os.path.abspath(HIST_DIR), f"{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return path


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")
