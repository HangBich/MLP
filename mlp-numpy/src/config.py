from dataclasses import dataclass, asdict, field
from typing import List


@dataclass
class Config:
    # --- Kiến trúc ---
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])  # 3 tầng ẩn
    activation: str = "relu"        # sigmoid | tanh | relu | leaky_relu
    init: str = "he"                # zeros | normal_small | normal_large | xavier | he
    use_bn: bool = False            # phần "Yêu cầu khác"
    leaky_slope: float = 0.01

    # --- Dữ liệu ---
    dataset: str = "fashion_mnist"  # fashion_mnist | mnist
    preprocess: str = "standardize"  # raw | scale01 | standardize
    val_ratio: float = 0.1

    # --- Huấn luyện ---
    lr: float = 0.1
    batch_size: int = 128
    epochs: int = 30
    seed: int = 0

    # --- Ghi nhận ---
    exp_group: str = "baseline"     # nhãn nhóm thí nghiệm, dùng để lọc CSV
    monitor_every: int = 1          # số epoch giữa 2 lần thu thống kê activation/gradient
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def run_id(self) -> str:
        """Định danh duy nhất cho lượt chạy -> dùng làm tên file kết quả (kèm seed)."""
        return (f"{self.exp_group}__act-{self.activation}__init-{self.init}"
                f"__prep-{self.preprocess}__bn-{int(self.use_bn)}"
                f"__lr-{self.lr}__bs-{self.batch_size}__seed-{self.seed}")
