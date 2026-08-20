# Đề tài 1 — Mạng nơ-ron từ số 0: MLP bằng NumPy

Bài tập lớn môn Học sâu — Trường CNTT&TT, ĐHBK Hà Nội.

Cài đặt MLP (≥ 3 tầng ẩn) hoàn toàn bằng NumPy (forward, backward, minibatch SGD,
không dùng autograd) trên Fashion-MNIST, kèm khảo sát có kiểm soát về hàm kích hoạt,
cách khởi tạo trọng số và cách tiền xử lý dữ liệu.

## Thành viên

| Họ tên | MSSV |Đóng góp |
|---|---|---|---|
|Nguyễn Thị Bích Hằng|20261098M|100% |

## Cài đặt

```bash
git clone https://github.com/HangBich/MLP/tree/main
cd mlp-numpy
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Yêu cầu: Python ≥ 3.10. CPU là đủ. Dữ liệu Fashion-MNIST tự tải về `./data/` ở lần chạy đầu.

## Chạy lại kết quả

Toàn bộ (≈ 39 lượt chạy):

```bash
bash reproduce.sh
```

Một lượt chạy đơn lẻ:

```bash
python -m experiments.run_experiment \
    --activation relu --init he --preprocess standardize \
    --hidden-sizes 256,128,64 --lr 0.1 --epochs 30 --seed 0 \
    --exp-group act_sweep
```

Bằng chứng gradient checking:

```bash
python -m experiments.run_gradcheck > results/gradcheck_report.md
```

Đầu ra:

| Đường dẫn | Nội dung |
|---|---|
| `logs/runs.csv` | Nhật ký thí nghiệm — mỗi dòng một lượt chạy |
| `results/histories/*.json` | Đường cong loss/acc + thống kê activation/gradient theo tầng |
| `results/figures/*.png` | Biểu đồ cho báo cáo |
| `results/gradcheck_report.md` | Bảng sai số tương đối gradient |
| `results/summary_tables.md` | Bảng mean ± std theo nhóm thí nghiệm |

## Cấu trúc thư mục

```
src/
  config.py        Dataclass Config — mọi siêu tham số của một lượt chạy
  data.py          Tải Fashion-MNIST, 3 chế độ tiền xử lý, minibatch iterator
  activations.py   Sigmoid / Tanh / ReLU / Leaky ReLU (forward + backward)
  initializers.py  zeros / normal_small / normal_large / Xavier / He
  layers.py        Linear, BatchNorm1D (tự cài cả forward lẫn backward)
  losses.py        Softmax + Cross-Entropy (gộp, ổn định số)
  model.py         Lớp MLP: forward, backward, params_and_grads
  optim.py         Minibatch SGD
  gradcheck.py     Numerical gradient checking (sai phân trung tâm)
  monitor.py       Thu thống kê activation/gradient theo từng tầng
  train.py         Vòng lặp huấn luyện cho một cấu hình
  utils.py         Seed, đo thời gian, ghi CSV, lưu history
experiments/
  run_experiment.py  CLI — điểm vào duy nhất cho mọi lượt chạy
  run_gradcheck.py   Sinh bằng chứng gradient checking
  torch_reference.py Đối chiếu với bản PyTorch tương đương
  aggregate.py       Gộp CSV thành bảng mean ± std
  make_plots.py      Vẽ toàn bộ biểu đồ
```

## Tái lập kết quả

- Mọi phép ngẫu nhiên đi qua `np.random.Generator` được tạo từ `set_seed(cfg.seed)`
  (khởi tạo trọng số **và** thứ tự shuffle minibatch).
- Seed nằm trong tên file kết quả và trong mỗi dòng `logs/runs.csv`.
- Mỗi cấu hình chạy 3 seed (0, 1, 2); báo cáo trung bình ± độ lệch chuẩn.
- Mỗi dòng log ghi kèm git commit hash.

## Nguồn tham khảo

> Ghi rõ mọi mã nguồn/tài liệu đã tham khảo, và nêu rõ phần nào nhóm tự viết.
> Sao chép không trích dẫn bị xử lý theo quy chế đạo văn.

- [ ] ...

**Phần nhóm tự viết:** toàn bộ `src/` (…điền cụ thể…).
**Phần kế thừa:** …

## Khai báo sử dụng công cụ AI

Xem mục tương ứng ở cuối báo cáo (`report/outline.md`).
# MLP
