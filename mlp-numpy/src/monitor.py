"""Thu thập thống kê activation và gradient THEO TỪNG TẦNG.

Đây là dữ liệu cho các biểu đồ bắt buộc:
  - phân bố activation theo tầng (4 hàm kích hoạt)
  - phân bố gradient theo tầng
  - minh họa bão hòa (sigmoid/tanh) và chết ReLU

!!! Thiết kế phần này TRƯỚC khi chạy loạt thí nghiệm. Nếu chạy 30 lượt
rồi mới nhận ra chưa lưu thống kê thì phải chạy lại toàn bộ.

Đừng lưu toàn bộ tensor (rất nặng). Mỗi tầng mỗi lần thu chỉ lưu:
    mean, std, các phân vị (5/25/50/75/95), tỉ lệ neuron chết (|a| < 1e-8),
    tỉ lệ bão hòa (|a| > 0.99 với tanh, hoặc a > 0.99 / a < 0.01 với sigmoid),
    histogram đã gộp bin (vd 50 bin, lưu counts + edges)
"""
import numpy as np


def summarize(arr, n_bins: int = 50) -> dict:
    """Nén một tensor thành dict thống kê nhẹ (JSON-serializable)."""
    raise NotImplementedError


class Monitor:
    """Bật/tắt việc thu thống kê. Model kiểm tra self.monitor.enabled trong
    forward/backward và gọi record() khi cần."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.records = []   # [{"epoch": e, "layer": i, "kind": "activation"|"gradient", "stats": {...}}]

    def record(self, epoch: int, layer_idx: int, kind: str, arr) -> None:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"records": self.records}
