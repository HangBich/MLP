# Checklist nộp bài — Đề tài 1

## Cài đặt lõi
- [ ] MLP ≥ 3 tầng ẩn, thuần NumPy (không autograd)
- [ ] Forward
- [ ] Backward
- [ ] Cập nhật tham số + minibatch SGD
- [ ] Gradient checking, sai số tương đối < 1e-5, **có bằng chứng trong báo cáo**

## Thí nghiệm bắt buộc
- [ ] 4 hàm kích hoạt: sigmoid, tanh, relu, leaky_relu
- [ ] Biểu đồ phân bố **activation** theo tầng
- [ ] Biểu đồ phân bố **gradient** theo tầng
- [ ] 5 cách khởi tạo: zeros, N(0,0.01), N(0,1), Xavier, He
- [ ] Giải thích bão hòa / chết ReLU quan sát được
- [ ] 3 cách tiền xử lý: raw, [0,1], zero-mean+unit-var
- [ ] Bảng accuracy + thời gian hội tụ của **tất cả** cấu hình
- [ ] ≥ 6 biểu đồ phân tích

## Yêu cầu khác (điểm sáng tạo)
- [ ] BatchNorm bằng NumPy — cả forward và backward
- [ ] Đo lại độ nhạy learning rate khi có BN
- [ ] Đối chiếu với bản PyTorch tương đương

## Sản phẩm nộp
- [ ] Repo GitHub / .zip
- [ ] README (cài đặt, chạy lại, trích dẫn, phần nhóm tự viết)
- [ ] requirements.txt
- [ ] Notebook/script tái lập kết quả chính
- [ ] Báo cáo PDF 8–12 trang, đúng 7 mục Phụ lục A
- [ ] Slide 8–12 trang
- [ ] Bảng phân công (tổng 100%)
- [ ] Nhật ký thí nghiệm CSV (mọi lượt chạy)
- [ ] Mục "Khai báo sử dụng công cụ AI"
- [ ] (Cộng điểm) video demo 3–5 phút hoặc web Gradio/Streamlit

## Chống mất điểm
- [ ] Clone repo mới + venv mới, chạy theo README → ra đúng kết quả (−15 nếu hỏng)
- [ ] Log đầy đủ (−10 nếu thiếu)
- [ ] Số liệu báo cáo khớp log (lệch = gian lận)
- [ ] Mọi hình/bảng có chú thích và được nhắc trong nội dung
- [ ] Ba người hỏi chéo phần của nhau trước buổi bảo vệ
