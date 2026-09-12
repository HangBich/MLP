# Mạng nơ-ron từ số 0 — MLP bằng NumPy thuần

Bài tập lớn môn **Học sâu và Ứng dụng (IT6130)** — Trường Công nghệ Thông tin và Truyền thông, Đại học Bách khoa Hà Nội.

Cài đặt mạng nơ-ron nhiều lớp (3 tầng ẩn) hoàn toàn bằng NumPy: tự viết lan truyền xuôi, lan truyền ngược và minibatch SGD, **không dùng cơ chế tự động vi phân** của bất kỳ thư viện học sâu nào. Bài toán hạ nguồn là phân loại ảnh quần áo trên Fashion-MNIST (ảnh xám 28×28, 10 lớp).

Ngoài phần cài đặt, repo còn chứa một khảo sát có kiểm soát theo **thiết kế delta**: xác lập một mô hình cơ sở, sau đó mỗi thí nghiệm chỉ thay đổi đúng một thành phần (hàm kích hoạt / cách khởi tạo / cách tiền xử lý) và so trực tiếp với mô hình cơ sở đó.

---

## Kết quả chính

| Hạng mục | Kết quả |
|---|---|
| Mô hình cơ sở | ReLU + He + standardize — **0.8847** trên test |
| Cấu hình tốt nhất | ReLU + He + **scale01** — **0.8889** trên test (+0.0042) |
| Gradient checking | sai số tương đối **1e−10** trên mọi tầng (ngưỡng yêu cầu 1e−5) |
| Đối chiếu PyTorch | lệch **~1e−16** ở dW, **3.3e−15** ở logits |
| Quy mô khảo sát | 13 cấu hình × 3 seed |
| Thời gian | 90,7 giây / 30 epoch (CPU một luồng); suy luận 0,0191 ms/ảnh |

Ba cấu hình hỏng hoàn toàn (`zeros`, `normal_large`, `raw`) đều cho accuracy đúng 0.1000 nhưng do **ba cơ chế khác nhau**: symmetry breaking failure, bùng nổ số học, và chết ReLU hàng loạt. Chi tiết trong báo cáo.

---

## Thành viên

| Họ tên | MSSV | Phần phụ trách | Đóng góp |
|---|---|---|---|
| Nguyễn Thị Bích Hằng | 20261098M | Code, báo cáo | 100% |

Giảng viên hướng dẫn: TS. Đặng Tuấn Linh

---

## Cài đặt

```bash
git clone <URL>
cd mlp-numpy

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Yêu cầu: **Python ≥ 3.10**. CPU là đủ, không cần GPU.
Fashion-MNIST tự tải về `./data/` ở lần chạy đầu tiên (4 file IDX, ~30 MB).

Để kết quả đo thời gian khớp với báo cáo, khóa số luồng BLAS về 1:

```bash
export OMP_NUM_THREADS=1
```

---

## Chạy demo

Cách nhanh nhất để kiểm tra repo hoạt động. Huấn luyện cấu hình tốt nhất rồi dự đoán trên 8 ảnh test lấy ngẫu nhiên, in nhãn thật / nhãn dự đoán / độ tin cậy và lưu hình minh họa:

```bash
python -m experiments.demo_predict \
    --activation relu --init he --preprocess scale01 \
    --epochs 30 --seed 0 \
    --num-samples 8 \
    --save results/figures/demo_predictions.png
```

Muốn xem kết quả ngay trong ~30 giây thì giảm số epoch (accuracy sẽ thấp hơn, chỉ để kiểm tra đường chạy):

```bash
python -m experiments.demo_predict --epochs 3 --num-samples 8
```

Nếu đã có trọng số lưu sẵn thì bỏ qua bước huấn luyện:

```bash
python -m experiments.demo_predict \
    --checkpoint results/checkpoints/relu_he_scale01_seed0.npz \
    --num-samples 12
```

Đầu ra mẫu:

```
Đang tải Fashion-MNIST ... 54000 train / 6000 val / 10000 test
Cấu hình: relu + he + scale01 | lr 0.1 | batch 128 | 30 epoch | seed 0
Huấn luyện ... hoàn tất trong 88.4 giây
Accuracy trên test: 0.8889

#  Nhãn thật        Dự đoán          Tin cậy   Kết quả
1  Áo phông         Áo phông          100%     đúng
2  Giày thể thao    Giày thể thao     100%     đúng
3  Bốt cổ ngắn      Bốt cổ ngắn       100%     đúng
4  Áo phông         Áo sơ mi           99%     SAI
5  Áo khoác         Áo khoác          100%     đúng
6  Túi xách         Túi xách          100%     đúng
7  Dép              Dép               100%     đúng
8  Dép              Dép               100%     đúng

Đúng 7/8. Đã lưu hình: results/figures/demo_predictions.png
```

Trường hợp sai ở trên là điển hình: mô hình nhầm áo phông với áo sơ mi — lớp gần nhất về hình dạng — chứ không nhầm với giày hay túi xách. Ở độ phân giải 28×28 thang xám, chi tiết phân biệt hai lớp này gần như biến mất.

---

## Chạy test

```bash
pytest -q tests/
```

| File | Nội dung kiểm tra |
|---|---|
| `tests/test_gradcheck.py` | Gradient giải tích khớp sai phân trung tâm, sai số < 1e−5 ở mọi tầng và mọi hàm kích hoạt |
| `tests/test_shapes.py` | Shape của forward/backward đúng qua mọi tầng, với batch lẻ và batch = 1 |
| `tests/test_softmax_stability.py` | Softmax không tràn với logits cỡ 1e3; loss không ra `nan` |
| `tests/test_reproducibility.py` | Cùng seed → cùng loss đến từng chữ số thập phân |
| `tests/test_demo.py` | **Chạy demo đầu-cuối 1 epoch**: tải dữ liệu → huấn luyện → dự đoán → sinh file hình, khẳng định accuracy > 0.7 và file đầu ra tồn tại |

Chạy riêng phần demo (nhanh, ~20 giây):

```bash
pytest -q tests/test_demo.py -s
```

Cờ `-s` để thấy trực tiếp bảng dự đoán mà demo in ra.

Chạy riêng gradient checking và xuất báo cáo:

```bash
python -m experiments.run_gradcheck > results/gradcheck_report.md
```

Đối chiếu với một cài đặt PyTorch tương đương (dựng `nn.Sequential` cùng kiến trúc, sao chép trọng số sang, cho cùng một batch đi qua cả hai):

```bash
python -m experiments.torch_reference
```

> PyTorch chỉ dùng để **đối chiếu**, không tham gia vào bất kỳ phần cài đặt nào của mô hình. Nếu chưa cài PyTorch thì bỏ qua lệnh này, mọi phần còn lại vẫn chạy bình thường.

---

## Chạy lại toàn bộ thí nghiệm

```bash
export OMP_NUM_THREADS=1
bash reproduce.sh
```

Khoảng **60 phút** trên CPU một luồng. Script chạy tuần tự 4 nhóm thí nghiệm, mỗi cấu hình 3 seed, rồi tự gộp bảng và vẽ biểu đồ.

Một lượt chạy đơn lẻ:

```bash
python -m experiments.run_experiment \
    --activation relu --init he --preprocess standardize \
    --hidden-sizes 256,128,64 \
    --lr 0.1 --batch-size 128 --epochs 30 --seed 0 \
    --exp-group act_sweep
```

Tham số CLI:

| Tham số | Giá trị hợp lệ | Mặc định |
|---|---|---|
| `--activation` | `relu` · `leaky_relu` · `tanh` · `sigmoid` | `relu` |
| `--init` | `he` · `xavier` · `normal_small` · `normal_large` · `zeros` | `he` |
| `--preprocess` | `standardize` · `scale01` · `raw` | `standardize` |
| `--hidden-sizes` | danh sách ngăn cách bởi dấu phẩy | `256,128,64` |
| `--lr` | số thực | `0.1` |
| `--batch-size` | số nguyên | `128` |
| `--epochs` | số nguyên | `30` |
| `--seed` | số nguyên | `0` |
| `--val-ratio` | số thực trong (0, 1) | `0.1` |
| `--exp-group` | tên nhóm, dùng để gộp bảng | `default` |

Gộp bảng và vẽ lại biểu đồ sau khi đã có log:

```bash
python -m experiments.aggregate > results/summary_tables.md
python -m experiments.make_plots
```

---

## Đầu ra

| Đường dẫn | Nội dung |
|---|---|
| `logs/runs.csv` | Nhật ký thí nghiệm — mỗi dòng một lượt chạy, kèm cấu hình, seed, thời gian, git commit hash |
| `results/histories/*.json` | Đường cong loss/accuracy theo epoch + thống kê activation/gradient theo tầng |
| `results/figures/*.png` | Toàn bộ biểu đồ dùng trong báo cáo |
| `results/checkpoints/*.npz` | Trọng số cuối của từng lượt chạy |
| `results/gradcheck_report.md` | Bảng sai số tương đối của gradient checking |
| `results/torch_reference.txt` | Kết quả đối chiếu với PyTorch |
| `results/summary_tables.md` | Bảng mean ± std theo từng nhóm thí nghiệm |

---

## Cấu trúc thư mục

```
src/
  config.py          Dataclass Config — mọi siêu tham số của một lượt chạy + run_id()
  data.py            Tải Fashion-MNIST, 3 chế độ tiền xử lý, minibatch iterator
  activations.py     Sigmoid / Tanh / ReLU / Leaky ReLU (forward + backward)
  initializers.py    zeros / normal_small / normal_large / Xavier / He
  layers.py          Linear, BatchNorm1D (tự cài cả forward lẫn backward)
  losses.py          Softmax + Cross-Entropy (gộp làm một, ổn định số học)
  model.py           Lớp MLP: forward, backward, params_and_grads
  optim.py           Minibatch SGD
  gradcheck.py       Numerical gradient checking (sai phân trung tâm)
  monitor.py         Thu thống kê activation/gradient theo từng tầng
  train.py           Vòng lặp huấn luyện cho một cấu hình
  utils.py           Seed, đo thời gian, ghi CSV, lưu history

experiments/
  run_experiment.py  CLI — điểm vào duy nhất cho mọi lượt chạy
  demo_predict.py    Demo: huấn luyện (hoặc nạp checkpoint) rồi dự đoán ảnh test
  run_gradcheck.py   Sinh bằng chứng gradient checking
  torch_reference.py Đối chiếu với bản PyTorch tương đương
  aggregate.py       Gộp CSV thành bảng mean ± std
  make_plots.py      Vẽ toàn bộ biểu đồ

tests/               pytest — xem mục "Chạy test"
```

Mọi tầng tuân theo cùng một giao diện gồm ba phương thức `forward`, `backward`, `params_and_grads`. Nhờ vậy lớp `MLP` chỉ cần một vòng lặp xuôi và một vòng lặp ngược, không phụ thuộc số tầng hay loại tầng.

---

## Thiết kế thí nghiệm

**Mô hình cơ sở:** ReLU + khởi tạo He + standardize, lr 0.1, batch 128, 30 epoch, seed {0, 1, 2}.

| Thí nghiệm | Thành phần thay đổi | Các giá trị khảo sát | Giữ nguyên |
|---|---|---|---|
| TN 1 | Hàm kích hoạt của 3 tầng ẩn | relu · leaky_relu · tanh · sigmoid | He, standardize |
| TN 2 | Sơ đồ khởi tạo trọng số | he · xavier · normal_small · normal_large · zeros | ReLU, standardize |
| TN 3 | Khởi tạo × hàm kích hoạt | 4 hàm kích hoạt chạy lại với Xavier | standardize |
| TN 4 | Chế độ tiền xử lý | standardize · scale01 · raw | ReLU, He |

Mỗi thí nghiệm giữ nguyên toàn bộ mô hình cơ sở và chỉ thay đổi đúng một thành phần, nên mọi chênh lệch quan sát được đều quy về đúng thay đổi đã thực hiện.

Độ lệch giữa các seed vào khoảng ±0.005 trên validation — chênh lệch nhỏ hơn mức này cần đọc thận trọng.

---

## Tái lập kết quả

- Mọi phép ngẫu nhiên đi qua một `np.random.Generator` duy nhất, tạo từ `set_seed(cfg.seed)` ở đầu hàm huấn luyện rồi truyền tường minh xuống: khởi tạo trọng số cả bốn tầng, chia train/validation, và thứ tự xáo trộn minibatch mỗi epoch.
- **Không dùng `np.random` toàn cục** — bộ sinh toàn cục chia sẻ trạng thái cho cả chương trình nên chỉ cần đổi thứ tự gọi hàm ở bất kỳ đâu là kết quả thay đổi theo.
- Seed nằm trong `run_id()`, tức nằm trong tên file kết quả và trong mỗi dòng `logs/runs.csv`.
- Mỗi cấu hình chạy 3 seed (0, 1, 2); mọi con số trong báo cáo là trung bình ± độ lệch chuẩn.
- Mỗi dòng log ghi kèm git commit hash của lúc chạy.

Chạy lại cùng một seed luôn cho cùng con số đến từng chữ số thập phân — `tests/test_reproducibility.py` kiểm tra đúng điều này.

---

## Hạn chế đã biết

- `BatchNorm1D` đã cài nhưng chưa có thí nghiệm đo độ nhạy với learning rate khi bật chuẩn hóa theo batch.
- Learning rate cố định 0.1 cho mọi cấu hình. Sigmoid có thể thu hẹp khoảng cách nếu được dò learning rate riêng, nhưng khi đó phép so sánh không còn cùng ngân sách.
- Thống kê activation/gradient lấy trên batch cuối của mỗi epoch chứ không phải trung bình toàn epoch.
- Trần accuracy ~0.89 là giới hạn của kiến trúc, không phải của siêu tham số: MLP duỗi ảnh thành vector 784 chiều nên mất hoàn toàn thông tin không gian. Muốn vượt trần này cần đổi sang mạng tích chập.

---

## Tài liệu tham khảo

1. Glorot & Bengio (2010). *Understanding the difficulty of training deep feedforward neural networks.* AISTATS.
2. He et al. (2015). *Delving deep into rectifiers.* ICCV.
3. Maas et al. (2013). *Rectifier nonlinearities improve neural network acoustic models.* ICML Workshop.
4. Xiao et al. (2017). *Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms.* arXiv:1708.07747.
5. Goodfellow, Bengio & Courville (2016). *Deep Learning.* MIT Press, chương 6 và 8.