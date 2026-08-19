#!/usr/bin/env bash
# Tái lập TOÀN BỘ kết quả chính. Chạy từ thư mục gốc của repo.
#   bash reproduce.sh
# Ước tính: ~39 lượt chạy x 15-30 phút CPU. Chia cho 3 máy bằng cách
# chạy mỗi máy một khối SEEDS/nhóm khác nhau.
set -euo pipefail

SEEDS="0 1 2"
RUN="python -m experiments.run_experiment"

echo "=== 0. Gradient checking (bằng chứng cho báo cáo) ==="
python -m experiments.run_gradcheck > results/gradcheck_report.md

echo "=== 1. Baseline ==="
for s in $SEEDS; do
  $RUN --exp-group baseline --activation relu --init he --preprocess standardize --seed $s
done

echo "=== 2. Hàm kích hoạt (4 cấu hình) ==="
for act in sigmoid tanh relu leaky_relu; do
  for s in $SEEDS; do
    $RUN --exp-group act_sweep --activation $act --init xavier --preprocess standardize --seed $s
  done
done

echo "=== 3. Khởi tạo (5 cấu hình) ==="
for ini in zeros normal_small normal_large xavier he; do
  for s in $SEEDS; do
    $RUN --exp-group init_sweep --activation relu --init $ini --preprocess standardize --seed $s
  done
done

echo "=== 4. Tiền xử lý (3 cấu hình) ==="
for prep in raw scale01 standardize; do
  for s in $SEEDS; do
    $RUN --exp-group prep_sweep --activation relu --init he --preprocess $prep --seed $s
  done
done

echo "=== 5. (Yêu cầu khác) BatchNorm: độ nhạy learning rate ==="
for lr in 0.001 0.01 0.1 0.5 1.0; do
  for s in $SEEDS; do
    $RUN --exp-group bn_lr --activation relu --init he --preprocess standardize --lr $lr --seed $s
    $RUN --exp-group bn_lr --activation relu --init he --preprocess standardize --lr $lr --seed $s --use-bn
  done
done

echo "=== 6. Đối chiếu PyTorch ==="
python -m experiments.torch_reference

echo "=== 7. Bảng tổng hợp + biểu đồ ==="
python -m experiments.aggregate > results/summary_tables.md
python -m experiments.make_plots

echo "Xong. Kết quả: logs/runs.csv, results/figures/, results/summary_tables.md"
