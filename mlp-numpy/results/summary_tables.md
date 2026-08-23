| exp_group   | activation   | init         | preprocess   |   val_acc_mean |   val_acc_std |   test_acc_mean |   test_acc_std |   conv_epochs |   time_sec |   n_seeds |
|:------------|:-------------|:-------------|:-------------|---------------:|--------------:|----------------:|---------------:|--------------:|-----------:|----------:|
| act_sweep   | leaky_relu   | xavier       | standardize  |         0.8904 |        0.0015 |          0.8887 |         0.005  |        1      |   108.39   |         3 |
| act_sweep   | relu         | xavier       | standardize  |         0.8901 |        0.0021 |          0.886  |         0.0053 |        1      |    93.9767 |         3 |
| act_sweep   | sigmoid      | xavier       | standardize  |         0.8854 |        0.0027 |          0.8753 |         0.0012 |        5      |   127.43   |         3 |
| act_sweep   | tanh         | xavier       | standardize  |         0.8905 |        0.0032 |          0.8836 |         0.0006 |        1      |    96.55   |         3 |
| baseline    | relu         | he           | standardize  |         0.8906 |        0.0049 |          0.8847 |         0.0009 |        1      |    90.6833 |         3 |
| init_sweep  | relu         | he           | standardize  |         0.8906 |        0.0049 |          0.8847 |         0.0009 |        1      |    92.38   |         3 |
| init_sweep  | relu         | normal_large | standardize  |         0.1004 |        0.0049 |          0.1    |         0      |        1      |    91.4633 |         3 |
| init_sweep  | relu         | normal_small | standardize  |         0.8892 |        0.0022 |          0.8834 |         0.0073 |        3.3333 |    94.7667 |         3 |
| init_sweep  | relu         | xavier       | standardize  |         0.8901 |        0.0021 |          0.886  |         0.0053 |        1      |    95.4033 |         3 |
| init_sweep  | relu         | zeros        | standardize  |         0.0986 |        0.0019 |          0.1    |         0      |        1.3333 |    83.24   |         3 |
| prep_sweep  | relu         | he           | raw          |         0.0951 |        0.0041 |          0.1    |         0      |        1.3333 |    82.0267 |         3 |
| prep_sweep  | relu         | he           | scale01      |         0.893  |        0.0037 |          0.8889 |         0.0011 |        1      |    87.03   |         3 |
| prep_sweep  | relu         | he           | standardize  |         0.8906 |        0.0049 |          0.8847 |         0.0009 |        1      |    89.12   |         3 |
