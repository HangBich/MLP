# Khung báo cáo (PDF 8–12 trang, không tính phụ lục)

Theo Phụ lục A của đề bài. Số trang là gợi ý phân bổ.

## 1. Giới thiệu (0,5 trang)
- Bài toán: phân loại ảnh Fashion-MNIST bằng MLP tự cài.
- Động cơ: hiểu forward/backward ở mức công thức, định lượng ảnh hưởng của
  hàm kích hoạt / khởi tạo / tiền xử lý.
- Đóng góp của nhóm: liệt kê 3–4 gạch đầu dòng cụ thể.

## 2. Cơ sở lý thuyết (1–2 trang) — có trích dẫn
- Forward/backward propagation, chain rule dạng ma trận.
- Softmax + cross-entropy, vì sao gộp hai bước.
- Hàm kích hoạt và đạo hàm; hiện tượng bão hòa, chết ReLU.
- Xavier/He: xuất phát từ điều kiện giữ phương sai qua các tầng.
- BatchNorm (nếu làm phần nâng cao).

## 3. Dữ liệu (1 trang)
- Fashion-MNIST: 60k train / 10k test, ảnh 28×28 xám, 10 lớp.
- Cách chia train/val (tỉ lệ, seed), phân bố lớp.
- Ba chế độ tiền xử lý; nêu rõ mean/std tính trên tập train.

## 4. Phương pháp (2–3 trang)
- Kiến trúc: 784 → 256 → 128 → 64 → 10; bảng số tham số từng tầng.
- Công thức backward viết ra tường minh (đây là phần lõi của đề tài này).
- **Gradient checking**: công thức sai phân trung tâm, eps, bảng sai số
  tương đối cho mọi tầng → bằng chứng < 1e-5. *(Bắt buộc có trong báo cáo.)*
- Siêu tham số cố định và siêu tham số được quét.

## 5. Thí nghiệm và kết quả (2–3 trang)
- Thiết lập: cùng seed, cùng ngân sách epoch, 3 seed/cấu hình.
- Bảng: accuracy + thời gian hội tụ của **tất cả** cấu hình (mean ± std).
- Biểu đồ (≥ 6) — xem danh sách trong `experiments/make_plots.py`.

## 6. Phân tích và thảo luận (1–2 trang)
Đây là mục ăn điểm (25đ). Không liệt kê số, phải **giải thích vì sao**:
- Khởi tạo toàn 0 → mọi neuron cùng tầng nhận gradient giống hệt nhau,
  không bao giờ phân hóa (symmetry breaking).
- Gaussian lớn (std=1.0) + sigmoid/tanh → pre-activation dạt ra đuôi,
  đạo hàm ≈ 0 → bão hòa; đối chiếu với fig phân bố activation.
- ReLU + lr lớn → tỉ lệ neuron chết tăng theo epoch; số liệu ở fig5.
- Vanishing gradient: chuẩn gradient giảm dần theo tầng ngược lên.
- Raw [0,255] → pre-activation rất lớn ngay từ đầu, cần lr nhỏ hơn nhiều.
- Khác biệt nào **nằm trong nhiễu** (nhỏ hơn độ lệch chuẩn) — phải nói rõ.
- Hạn chế của thí nghiệm.

## 7. Kết luận và hướng phát triển (0,5 trang)

## Tài liệu tham khảo

## Phụ lục
- Bảng phân công (tổng 100%).
- **Khai báo sử dụng công cụ AI**: dùng công cụ nào, cho việc gì.
- Hình ảnh bổ sung.
