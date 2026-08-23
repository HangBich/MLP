import pandas as pd 

df = pd.read_csv("logs/runs.csv")
g = (df.groupby(["exp_group", "activation", "init", "preprocess"])
     .agg(
         val_acc_mean=("val_acc", "mean"),
         val_acc_std=("val_acc", "std"),
         test_acc_mean=("test_acc", "mean"),
         test_acc_std=("test_acc", "std"),
         conv_epochs=("epochs_to_90pct_best", "mean"),
         time_sec=("wall_time_sec", "mean"),
         n_seeds=("seed", "count"))
      .round(4).reset_index())
df = df.drop_duplicates(subset=["run_id"], keep="last")
print(g.to_markdown(index=False))