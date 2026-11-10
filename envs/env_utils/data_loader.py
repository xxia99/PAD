from envs.env_utils import data_configs
import numpy as np
import pickle

def _synthetic_in_out_data(delta, sample_size, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    col_blocks = [data_configs.M_total // 5] * 5
    col_blocks[-1] += data_configs.M_total - sum(col_blocks)
    cols_normal, cols_t, cols_exp, cols_log, cols_poi = col_blocks

    test_in_data = np.zeros((sample_size, data_configs.M_total))
    test_out_data = np.zeros((sample_size, data_configs.M_total))
    start = 0

    if cols_normal > 0:
        in_block = rng.normal(size=(sample_size, cols_normal))
        out_block = rng.normal(loc=delta, size=(sample_size, cols_normal))
        test_in_data[:, start:start + cols_normal] = in_block
        test_out_data[:, start:start + cols_normal] = out_block
        start += cols_normal
    if cols_t > 0:
        in_block = data_configs.normalize_data(rng.standard_t(df=3, size=(sample_size, cols_t)))
        out_block = data_configs.normalize_data(rng.standard_t(df=3, size=(sample_size, cols_t))) + delta
        test_in_data[:, start:start + cols_t] = in_block
        test_out_data[:, start:start + cols_t] = out_block
        start += cols_t
    if cols_exp > 0:
        in_block = data_configs.normalize_data(rng.exponential(scale=1, size=(sample_size, cols_exp)))
        out_block = data_configs.normalize_data(rng.exponential(scale=1, size=(sample_size, cols_exp))) + delta
        test_in_data[:, start:start + cols_exp] = in_block
        test_out_data[:, start:start + cols_exp] = out_block
        start += cols_exp
    if cols_log > 0:
        in_block = data_configs.normalize_data(
            (rng.lognormal(mean=1, sigma=0.5, size=(sample_size, cols_log)) - cols_log) / 1.6
        )
        out_block = data_configs.normalize_data(
            (rng.lognormal(mean=1, sigma=0.5, size=(sample_size, cols_log)) - cols_log) / 1.6
        ) + delta
        test_in_data[:, start:start + cols_log] = in_block
        test_out_data[:, start:start + cols_log] = out_block
        start += cols_log
    if cols_poi > 0:
        in_block = data_configs.normalize_data((rng.poisson(lam=10, size=(sample_size, cols_poi)) - 10) / np.sqrt(10))
        out_block = data_configs.normalize_data((rng.poisson(lam=10, size=(sample_size, cols_poi)) - 10) / np.sqrt(10)) + delta
        test_in_data[:, start:start + cols_poi] = in_block
        test_out_data[:, start:start + cols_poi] = out_block

    return test_in_data, test_out_data


# assuming one data stream breaks
def one_episode_data(delta, k, broken_monitor):
    t = 50
    horizon = t + k
    train_in_data, train_out_data = _synthetic_in_out_data(delta=delta, sample_size=2000)
    data=np.zeros((data_configs.M_total,t+k))
    for monitor in range(data_configs.M_total):
        monitor_data=train_in_data[:,monitor]
        bootstrap_data=np.random.choice(monitor_data,horizon)
        data[monitor,0:t+k]=bootstrap_data
    out_monitor_data=train_out_data[:,broken_monitor]
    breakdown_data=np.random.choice(out_monitor_data,k)
    data[broken_monitor,t:t+k]=breakdown_data.T
    return data, t

def test_one_episode_data(delta,k,broken_monitors, seed=None):
    # t=np.random.randint(data_configs.time_range_start,data_configs.time_range_end)
    t = 50
    k -= t

    test_in_data_normal = np.random.normal(size=(1000, data_configs.M_total // 5))
    test_out_data_normal = np.random.normal(loc=delta, size=(1000, data_configs.M_total // 5))

    test_in_data_t = data_configs.normalize_data(np.random.standard_t(df=3, size=(1000, data_configs.M_total // 5)))
    test_out_data_t = data_configs.normalize_data(np.random.standard_t(df=3, size=(1000, data_configs.M_total // 5))) + np.ones((1000, data_configs.M_total // 5)) * delta

    test_in_data_exp = data_configs.normalize_data(np.random.exponential(scale=1, size=(1000, data_configs.M_total // 5)))
    test_out_data_exp = data_configs.normalize_data(np.random.exponential(scale=1, size=(1000, data_configs.M_total // 5))) + np.ones(
        (1000, data_configs.M_total // 5)) * delta

    test_in_data_log = data_configs.normalize_data((np.random.lognormal(mean=1, sigma=0.5, size=(1000, data_configs.M_total // 5)) - data_configs.M_total // 5) / 1.6)
    test_out_data_log = data_configs.normalize_data(
        (np.random.lognormal(mean=1, sigma=0.5, size=(1000, data_configs.M_total // 5)) - data_configs.M_total // 5) / 1.6) + np.ones((1000, data_configs.M_total // 5)) * delta

    test_in_data_poi = data_configs.normalize_data((np.random.poisson(lam=10, size=(1000, data_configs.M_total // 5)) - 10) / np.sqrt(10))
    test_out_data_poi = data_configs.normalize_data((np.random.poisson(lam=10, size=(1000, data_configs.M_total // 5)) - 10) / np.sqrt(10)) + np.ones(
        (1000, data_configs.M_total // 5)) * delta

    test_in_data = np.zeros((1000, data_configs.M_total))
    test_in_data[:, 0:data_configs.M_total // 5] = test_in_data_normal
    test_in_data[:, data_configs.M_total // 5:2 * data_configs.M_total // 5] = test_in_data_t
    test_in_data[:, 2 * data_configs.M_total // 5:3 * data_configs.M_total // 5] = test_in_data_exp
    test_in_data[:, 3 * data_configs.M_total // 5:4 * data_configs.M_total // 5] = test_in_data_log
    test_in_data[:, 4 * data_configs.M_total // 5:data_configs.M_total] = test_in_data_poi

    test_out_data = np.zeros((1000, data_configs.M_total))
    test_out_data[:, 0:data_configs.M_total // 5] = test_out_data_normal
    test_out_data[:, data_configs.M_total // 5:2 * data_configs.M_total // 5] = test_out_data_t
    test_out_data[:, 2 * data_configs.M_total // 5:3 * data_configs.M_total // 5] = test_out_data_exp
    test_out_data[:, 3 * data_configs.M_total // 5:4 * data_configs.M_total // 5] = test_out_data_log
    test_out_data[:, 4 * data_configs.M_total // 5:data_configs.M_total] = test_out_data_poi

    data=np.zeros((data_configs.M_total, t+k))
    for monitor in range(data_configs.M_total):
        monitor_data = test_in_data[:,monitor]
        bootstrap_data=np.random.choice(monitor_data,t+k)
        data[monitor, 0:t+k]=bootstrap_data
    if isinstance(broken_monitors, int):
        broken_monitors = [broken_monitors]

    for broken_monitor in broken_monitors:
        out_monitor_data = test_out_data[:,broken_monitor]
        breakdown_data = np.random.choice(out_monitor_data,k)
        data[broken_monitor, t:t+k] = breakdown_data.T
    return data, t

def eval_one_episode_data(delta, k, broken_monitor, seed):
    t = 10 #50
    rng = np.random.default_rng(seed)
    train_in_data, train_out_data = _synthetic_in_out_data(delta=delta, sample_size=2000, rng=rng)
    data=np.zeros((data_configs.M_total,t+k))
    for monitor in range(data_configs.M_total):
        monitor_data=train_in_data[:,monitor]
        bootstrap_data=rng.choice(monitor_data,t+k)
        data[monitor,0:t+k]=bootstrap_data
    out_monitor_data=train_out_data[:,broken_monitor]
    breakdown_data=rng.choice(out_monitor_data,k)
    data[broken_monitor,t:t+k]=breakdown_data.T
    return data, t

def case_one_episode_data(delta,k,broken_monitors, seed=None):
    import pandas as pd

    # t=np.random.randint(data_configs.time_range_start,data_configs.time_range_end)
    t = 50  # 50
    k -= t

    filename = "datasets/ic_test.csv"
    df = pd.read_csv(filename)
    data = df.iloc[:, 3:].values
    ic_data = data[:500, :data_configs.M_total]

    test_in_data = data_configs.normalize_data(ic_data)
    test_out_data = test_in_data + np.ones((1000, data_configs.M_total)) * delta

    data=np.zeros((data_configs.M_total, t+k))
    for monitor in range(data_configs.M_total):
        monitor_data = test_in_data[:,monitor]
        bootstrap_data=np.random.choice(monitor_data,t+k)
        data[monitor, 0:t+k]=bootstrap_data
    if isinstance(broken_monitors, int):
        broken_monitors = [broken_monitors]

    for broken_monitor in broken_monitors:
        out_monitor_data = test_out_data[:,broken_monitor]
        breakdown_data = np.random.choice(out_monitor_data,k)
        data[broken_monitor, t:t+k] = breakdown_data.T
    return data, t

def eval_case_one_episode_data(delta,k,broken_monitors, seed=None):
    import pandas as pd

    # t=np.random.randint(data_configs.time_range_start,data_configs.time_range_end)
    # np.random.seed(seed)
    t = 5  # 50
    k -= t

    filename = "datasets/ic_test.csv"
    df = pd.read_csv(filename)
    df = df[df["simulationRun"] == seed % 20 + 1]
    data = df.iloc[:, 3:].values
    ic_data = data[:, :data_configs.M_total]

    test_in_data = data_configs.normalize_data(ic_data)
    test_out_data = test_in_data + np.ones((ic_data.shape[0], data_configs.M_total)) * delta
    if test_in_data.shape[0] < t+k:
        print("test_in_data: ", test_in_data, "seed: ", seed)

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # figure = plt.figure(figsize=(10, 5))
    # sns.set(style="whitegrid")
    # (ax1, ax2) = figure.subplots(1, 2, sharex="col")

    data=np.zeros((data_configs.M_total, t+k))
    for monitor in range(data_configs.M_total):
        monitor_data = test_in_data[:, monitor]
        if monitor_data.shape[0] < t+k:
            print("data: ", monitor_data)
        data[monitor, :t+k] = monitor_data[:t+k]

    if isinstance(broken_monitors, int):
        broken_monitors = [broken_monitors]
    for broken_monitor in broken_monitors:
        # sns.histplot(data[broken_monitor, :], kde=True, ax=ax1)
        out_monitor_data = test_out_data[:, broken_monitor]
        data[broken_monitor, t:t+k] = out_monitor_data[:k]
        # sns.histplot(data[broken_monitor, :], kde=True, ax=ax2)
        # plt.title(f"Monitor {broken_monitor} Data Distribution")
        # plt.show()

    return data, t


def eval_pi_one_episode_data(delta,k,broken_monitors, seed=None):
    with open('datasets/IoT_case.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    data_ic = np.asarray(data_dict['data_ic'])
    if data_ic.shape[0] != data_configs.M_total:
        data_ic = data_ic.T
    data_ic = data_ic[:data_configs.M_total, :]
    row_mean = np.mean(data_ic, axis=1, keepdims=True)
    print("row mean shape: ", row_mean.shape)
    row_std = np.std(data_ic, axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0
    data_ic = (data_ic - row_mean) / row_std

    def _normalize_episode_batch(batch):
        # print("Normalizing episode batch with shape: ", batch.shape)
        batch = np.asarray(batch)
        for i in range(batch.shape[1]):
            monitor_mean = row_mean[i, 0]
            monitor_std = row_std[i, 0]
            batch[:, i, :] = (batch[:, i, :] - monitor_mean) / monitor_std
        return batch
        # if batch.shape[1] == data_configs.M_total:
        #     return (batch - row_mean[None, :, :]) / row_std[None, :, :]
        # if batch.shape[2] == data_configs.M_total:
        #     monitor_mean = row_mean[:, 0]
        #     monitor_std = row_std[:, 0]
        #     return (batch - monitor_mean[None, None, :]) / monitor_std[None, None, :]
        raise ValueError(
            f"Episode batch shape {batch.shape} does not align with M_total={data_configs.M_total}."
        )

    data_1 = np.asarray(data_dict.get('data_1', data_dict.get('data1')))
    data_2 = np.asarray(data_dict.get('data_2', data_dict.get('data2')))
    data_3 = np.asarray(data_dict.get('data_3', data_dict.get('data3')))
    data_1 = _normalize_episode_batch(data_1)
    data_2 = _normalize_episode_batch(data_2)
    data_3 = _normalize_episode_batch(data_3)

    # print("data_1 first 30 mean: ", np.mean(data_1[1, 0, :]))
    # print("data_1 after 30 mean: ", np.mean(data_1[1, 0, :]))

    t = 30
    rng = np.random.default_rng(seed)

    if float(delta) == 0.0:
        horizon = int(k)
        test_in_data = data_ic.T
        data = np.zeros((data_configs.M_total, horizon))
        for monitor in range(data_configs.M_total):
            monitor_data = test_in_data[:, monitor]
            data[monitor, :] = rng.choice(monitor_data, horizon, replace=True)
        return data, t

    shift_key = int(round(float(delta)))
    if shift_key not in {1, 2, 3}:
        raise ValueError(f"eval_pi mode supports mean shift in {{0,1,2,3}}, got {delta}.")

    selected = {1: data_1, 2: data_2, 3: data_3}[shift_key]
    episode_idx = int(seed or 0) % selected.shape[0]
    episode = np.asarray(selected[episode_idx])
    if episode.shape[0] != data_configs.M_total:
        episode = episode.T
    episode = episode[:data_configs.M_total, :]
    # print("Selected episode shape: ", episode.shape)

    broken_monitors = []
    for monitor in range(episode.shape[0]):
        # print(episode[monitor, 30:].shape)
        if np.mean(episode[monitor, 30:]) >= 0.5:
            broken_monitors.append(monitor)
            # print("broken monitor: ", monitor,
            #       "mean before 30: ", np.mean(episode[monitor, :30]),
            #       "mean after 30: ", np.mean(episode[monitor, 30:]))
    return episode, t, broken_monitors
