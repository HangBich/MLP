"""Gộp logs/runs.csv thành bảng mean ± std theo nhóm thí nghiệm,
xuất Markdown/LaTeX để dán thẳng vào báo cáo.

    python -m experiments.aggregate --group act_sweep

Nhóm theo (exp_group, activation, init, preprocess, use_bn), tổng hợp trên
cột seed. Báo cáo cả accuracy lẫn epochs_to_90pct_best (thời gian hội tụ).
Khác biệt nhỏ hơn độ lệch chuẩn thì phải ghi rõ là 'nằm trong nhiễu'.
"""
raise NotImplementedError
