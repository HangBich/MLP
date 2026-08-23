"""Vẽ toàn bộ biểu đồ cho báo cáo.

Chạy sau khi sweep xong và đã chạy collect_stats.py:
    python -m experiments.make_plots
"""
import glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = "results/figures"
STATS = "results/activation_gradient_stats.json"


def load_histories():
    out = {}
    for p in glob.glob("results/histories/*.json"):
        with open(p) as f:
            out[os.path.basename(p)[:-5]] = json.load(f)
    return out


def _curves(hists, prefix, key, fname, title_hint):
    """Vẽ loss + accuracy theo epoch cho một nhóm thí nghiệm."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for k, h in sorted(hists.items()):
        if not k.startswith(prefix) or "seed-0" not in k:
            continue
        name = k.split(key)[1].split("__")[0]
        ax1.plot(h["train_loss"], label=name)
        ax2.plot(h["val_acc"], label=name)
    ax1.set(xlabel="Epoch", ylabel="Train loss", title=f"Loss theo {title_hint}")
    ax2.set(xlabel="Epoch", ylabel="Val accuracy", title=f"Accuracy theo {title_hint}")
    for ax in (ax1, ax2):
        ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}", dpi=150); plt.close(fig)


def plot_fig1(h): _curves(h, "act_sweep",  "act-",  "fig1_activation_curves.png", "hàm kích hoạt")
def plot_fig4(h): _curves(h, "init_sweep", "init-", "fig4_init_curves.png",       "cách khởi tạo")
def plot_fig6(h): _curves(h, "prep_sweep", "prep-", "fig6_preprocess_curves.png", "cách tiền xử lý")


def plot_fig2(recs):
    """Phân bố activation theo tầng, lưới 4 activation x các tầng."""
    acts = ["sigmoid", "tanh", "relu", "leaky_relu"]
    last_ep = max(r["epoch"] for r in recs)
    layers = sorted({r["layer"] for r in recs if r["kind"] == "activation"})

    fig, axes = plt.subplots(len(acts), len(layers),
                             figsize=(3.2 * len(layers), 2.6 * len(acts)),
                             squeeze=False)
    for i, act in enumerate(acts):
        for j, L in enumerate(layers):
            ax = axes[i][j]
            for r in recs:
                if (r["kind"] == "activation" and r["act"] == act
                        and r["layer"] == L and r["epoch"] == last_ep):
                    ax.stairs(r["hist"], r["edges"], fill=True, alpha=.7)
            if i == 0: ax.set_title(f"Tầng {L}")
            if j == 0: ax.set_ylabel(act)
            ax.tick_params(labelsize=7)
    fig.suptitle("Phân bố activation theo tầng (batch cuối, epoch cuối)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_activation_dist.png", dpi=150); plt.close(fig)


def plot_fig3(recs):
    """Chuẩn gradient ||dW|| theo chỉ số tầng — minh họa vanishing gradient."""
    last_ep = max(r["epoch"] for r in recs)
    fig, ax = plt.subplots(figsize=(7, 4))
    for act in ["sigmoid", "tanh", "relu", "leaky_relu"]:
        pts = [(r["layer"], r["grad_norm"]) for r in recs
               if r["kind"] == "gradient" and r["act"] == act and r["epoch"] == last_ep]
        pts.sort()
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=act)
    ax.set(xlabel="Chỉ số tầng (0 = gần đầu vào)", ylabel="||dW||",
           title="Chuẩn gradient theo tầng", yscale="log")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_grad_norm.png", dpi=150); plt.close(fig)


def plot_fig5(recs):
    """Tỉ lệ neuron chết và tỉ lệ bão hòa theo epoch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for act in ["sigmoid", "tanh", "relu", "leaky_relu"]:
        rows = [r for r in recs if r["kind"] == "activation" and r["act"] == act]
        eps = sorted({r["epoch"] for r in rows})
        dead = [np.mean([r["dead_frac"] for r in rows if r["epoch"] == e]) for e in eps]
        sat  = [np.mean([r["sat_frac"]  for r in rows if r["epoch"] == e]) for e in eps]
        ax1.plot(eps, dead, "o-", label=act)
        ax2.plot(eps, sat, "o-", label=act)
    ax1.set(xlabel="Epoch", ylabel="Tỉ lệ |a| < 1e-8", title="Tỉ lệ neuron chết")
    ax2.set(xlabel="Epoch", ylabel="Tỉ lệ |a| > 0.99", title="Tỉ lệ bão hòa")
    for ax in (ax1, ax2):
        ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_dead_saturated.png", dpi=150); plt.close(fig)


def plot_fig7(df):
    """Bar chart val_acc mean ± std cho cả ba nhóm thí nghiệm."""
    groups = [("act_sweep", "activation", "Hàm kích hoạt"),
              ("init_sweep", "init", "Cách khởi tạo"),
              ("prep_sweep", "preprocess", "Cách tiền xử lý")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (grp, col, title) in zip(axes, groups):
        g = df[df.exp_group == grp].groupby(col)["val_acc"].agg(["mean", "std"])
        ax.bar(g.index, g["mean"], yerr=g["std"], capsize=4)
        ax.set(ylabel="Val accuracy", title=title, ylim=(0, 1))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=.3, axis="y")
    fig.suptitle("Accuracy trên tập validation (mean ± std, 3 seed)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig7_bars.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)

    hists = load_histories()
    plot_fig1(hists); plot_fig4(hists); plot_fig6(hists)

    df = pd.read_csv("logs/runs.csv").drop_duplicates("run_id", keep="last")
    plot_fig7(df)

    if os.path.exists(STATS):
        with open(STATS) as f:
            recs = json.load(f)
        plot_fig2(recs); plot_fig3(recs); plot_fig5(recs)
    else:
        print(f"Chưa có {STATS} — chạy collect_stats.py để có fig2/3/5")

    print("Đã lưu biểu đồ vào", FIG)
