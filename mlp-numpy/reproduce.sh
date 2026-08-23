set -euo pipefail

SEEDS="0 1 2"
RUN="python -m experiments.run_experiment"

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

echo "Xong. Kết quả: logs/runs.csv, results/figures/, results/summary_tables.md"
