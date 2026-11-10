# Dataset Placement

Synthetic experiments do not need external data.

For the IoT case study, place:

```text
datasets/IoT_case.pkl
```

Expected pickle keys:

```text
data_ic, data1, data2, data3
```

For the Tennessee Eastman Process case study, place:

```text
datasets/TEP/ic_train.csv
datasets/TEP/ic_test.csv
```

Expected columns:

```text
xmeas_1, ..., xmeas_41, xmv_1, ..., xmv_11
```

The TEP loader reorders columns into three PAD regions of sizes `22,16,14`.
