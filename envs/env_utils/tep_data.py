from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


# TEP variables are stored as XMEAS(1:41), XMV(1:11).  The order below makes
# each PAD region contiguous after reindexing.
TEP_REGION_COLUMNS = {
    "reactor_condenser": [
        "xmeas_1",
        "xmeas_2",
        "xmeas_3",
        "xmeas_4",
        "xmeas_6",
        "xmeas_7",
        "xmeas_8",
        "xmeas_9",
        "xmeas_21",
        "xmeas_22",
        "xmeas_23",
        "xmeas_24",
        "xmeas_25",
        "xmeas_26",
        "xmeas_27",
        "xmeas_28",
        "xmv_1",
        "xmv_2",
        "xmv_3",
        "xmv_4",
        "xmv_10",
        "xmv_11",
    ],
    "separator_compressor": [
        "xmeas_5",
        "xmeas_10",
        "xmeas_11",
        "xmeas_12",
        "xmeas_13",
        "xmeas_20",
        "xmeas_29",
        "xmeas_30",
        "xmeas_31",
        "xmeas_32",
        "xmeas_33",
        "xmeas_34",
        "xmeas_35",
        "xmeas_36",
        "xmv_5",
        "xmv_6",
    ],
    "stripper": [
        "xmeas_14",
        "xmeas_15",
        "xmeas_16",
        "xmeas_17",
        "xmeas_18",
        "xmeas_19",
        "xmeas_37",
        "xmeas_38",
        "xmeas_39",
        "xmeas_40",
        "xmeas_41",
        "xmv_7",
        "xmv_8",
        "xmv_9",
    ],
}


def tep_region_columns():
    return [list(cols) for cols in TEP_REGION_COLUMNS.values()]


def tep_ordered_columns():
    return [col for group in tep_region_columns() for col in group]


def tep_region_sizes():
    return [len(cols) for cols in tep_region_columns()]


def _resolve_tep_dir(path):
    base = Path(path or "datasets/TEP")
    if base.is_file():
        base = base.parent
    return base


def _read_tep_csv(base, filename):
    csv_path = base / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing TEP CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    ordered = tep_ordered_columns()
    missing = [col for col in ordered if col not in df.columns]
    if missing:
        raise ValueError(f"TEP CSV {csv_path} is missing columns: {missing}")
    return df.loc[:, ordered].to_numpy(dtype=np.float32, copy=True)


@lru_cache(maxsize=4)
def load_standardized_tep_arrays(base_path="datasets/TEP"):
    base = _resolve_tep_dir(base_path)
    train_raw = _read_tep_csv(base, "ic_train.csv")
    test_raw = _read_tep_csv(base, "ic_test.csv")
    mean = train_raw.mean(axis=0, keepdims=True)
    std = train_raw.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    train = ((train_raw - mean) / std).astype(np.float32, copy=False)
    test = ((test_raw - mean) / std).astype(np.float32, copy=False)
    return train, test, mean.astype(np.float32), std.astype(np.float32)
