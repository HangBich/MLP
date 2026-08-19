"""Numerical gradient checking — BẮT BUỘC, sai số tương đối < 1e-5,
và phải in ra được bằng chứng đưa vào báo cáo.

Công thức sai phân trung tâm:
    g_num[i] = (J(theta + eps*e_i) - J(theta - eps*e_i)) / (2*eps)

Sai số tương đối:
    rel = ||g_num - g_ana|| / (||g_num|| + ||g_ana|| + tiny)

Mẹo để check thành công:
  - Dùng float64 (mặc định của NumPy) — float32 không đủ chính xác.
  - eps ~ 1e-5 .. 1e-7.
  - Mạng NHỎ (vd 10 mẫu, tầng ẩn 5-4-3) và batch cố định.
  - TẮT mọi thứ ngẫu nhiên (dropout, shuffle) trong lúc check.
  - ReLU không khả vi tại 0: nếu rel error lớn bất thường, thử tanh trước
    để tách bạch lỗi cài đặt với lỗi kink của ReLU.
"""
import numpy as np


def relative_error(g_num, g_ana) -> float:
    raise NotImplementedError


def check_layer_gradients(model, X, y, eps: float = 1e-5, max_params_per_tensor: int = 50):
    """Kiểm tra gradient của MỌI tham số (W, b, gamma, beta).

    Với mỗi tensor, lấy mẫu ngẫu nhiên tối đa `max_params_per_tensor` phần tử
    (kiểm tra toàn bộ 784x256 phần tử là không cần thiết và rất chậm).

    Returns
    -------
    list[dict] : [{"layer": int, "param": "W", "rel_error": float, "passed": bool}, ...]
    """
    raise NotImplementedError


def format_report(results) -> str:
    """Xuất bảng text/Markdown để dán thẳng vào báo cáo.
    Nên có cột: Tầng | Tham số | Sai số tương đối | Đạt (<1e-5)."""
    raise NotImplementedError
