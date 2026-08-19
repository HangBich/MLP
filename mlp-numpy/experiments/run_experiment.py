"""Điểm vào duy nhất cho MỌI lượt chạy.

Không bao giờ chạy tay trong notebook rồi chép số sang báo cáo — số liệu
lệch với log bị xử như gian lận học thuật.

Ví dụ:
    python -m experiments.run_experiment --activation relu --init he \
        --preprocess standardize --seed 0 --exp-group act_sweep
"""
import argparse
import sys

from src.config import Config
from src.train import train_one_run
from src.utils import append_run, git_commit, now


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--activation", default="relu",
                   choices=["sigmoid", "tanh", "relu", "leaky_relu"])
    p.add_argument("--init", default="he",
                   choices=["zeros", "normal_small", "normal_large", "xavier", "he"])
    p.add_argument("--preprocess", default="standardize",
                   choices=["raw", "scale01", "standardize"])
    p.add_argument("--hidden-sizes", default="256,128,64",
                   help="Danh sách kích thước tầng ẩn, cách nhau bởi dấu phẩy (>= 3 tầng)")
    p.add_argument("--use-bn", action="store_true")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="fashion_mnist", choices=["fashion_mnist", "mnist"])
    p.add_argument("--exp-group", default="baseline")
    p.add_argument("--notes", default="")
    a = p.parse_args(argv)

    hidden = [int(x) for x in a.hidden_sizes.split(",") if x.strip()]
    assert len(hidden) >= 3, "Đề bài yêu cầu tối thiểu 3 tầng ẩn."

    return Config(
        hidden_sizes=hidden, activation=a.activation, init=a.init,
        use_bn=a.use_bn, dataset=a.dataset, preprocess=a.preprocess,
        lr=a.lr, batch_size=a.batch_size, epochs=a.epochs, seed=a.seed,
        exp_group=a.exp_group, notes=a.notes,
    )


def main(argv=None) -> int:
    cfg = parse_args(argv)
    result = train_one_run(cfg)

    append_run({
        "run_id": cfg.run_id(),
        "timestamp": now(),
        "exp_group": cfg.exp_group,
        "activation": cfg.activation,
        "init": cfg.init,
        "preprocess": cfg.preprocess,
        "hidden_sizes": "-".join(map(str, cfg.hidden_sizes)),
        "use_bn": int(cfg.use_bn),
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "seed": cfg.seed,
        "git_commit": git_commit(),
        "notes": cfg.notes,
        **result,
    })
    print(f"[done] {cfg.run_id()}  val_acc={result.get('val_acc')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
