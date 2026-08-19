"""Sinh bằng chứng gradient checking để dán vào báo cáo.

    python -m experiments.run_gradcheck > results/gradcheck_report.md

Chạy check trên mạng nhỏ, batch cố định, float64, cho CẢ 4 hàm kích hoạt
và cho cấu hình có BatchNorm (nếu đã cài).
"""
# TODO:
#   1. Dựng MLP nhỏ (vd hidden [5, 4, 3], input 20 chiều, 10 mẫu giả)
#   2. forward -> softmax_cross_entropy -> backward
#   3. check_layer_gradients(...) -> format_report(...)
#   4. In ra Markdown; lặp qua activation in ACTIVATIONS và use_bn in [False, True]
raise NotImplementedError
