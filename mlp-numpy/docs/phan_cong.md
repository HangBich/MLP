# Bảng phân công công việc

Tổng đóng góp phải bằng 100%. Điểm cá nhân = điểm nhóm × hệ số (0,7–1,1).
Thành viên không trả lời được câu hỏi về phần mình khai đã làm sẽ bị hạ hệ số.

| Thành viên | MSSV | Công việc | Sản phẩm cụ thể | Đóng góp |
|---|---|---|---|---|
| | | Lõi NumPy (cùng làm) + thí nghiệm hàm kích hoạt + monitor/biểu đồ | `activations.py`, `monitor.py`, `make_plots.py` | % |
| | | Lõi NumPy (cùng làm) + thí nghiệm khởi tạo & tiền xử lý | `initializers.py`, `data.py`, mục 6 báo cáo | % |
| | | Lõi NumPy (cùng làm) + BatchNorm + đối chiếu PyTorch | `layers.py` (BN), `torch_reference.py` | % |

## Lưu ý phân chia

Giai đoạn 1 (2–3 ngày đầu) — **cả ba cùng làm** lõi: `model.py`, `layers.py` (Linear),
`losses.py`, `gradcheck.py`. Khi bảo vệ, thầy có thể hỏi backprop cho bất kỳ ai,
không riêng người viết.

Giai đoạn 2 — chia ba chạy song song theo bảng trên.
