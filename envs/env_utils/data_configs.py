import numpy as np

# episode_number=1000 #threshold learning number of episode
u = 0.5
allowance = 0.5
l = 1
d = 10
epoch_buffer = 0  # 5
epsilon = 1
gamma = 0.99

DEFAULT_M_TOTAL = None
DEFAULT_M_TOTAL_RESOURCES = None  # 4   # 10 # 20
DEFAULT_AGENT_NUM = None
DEFAULT_DELTA = None
use_torch_stats = False

np.random.seed(1)


def _parse_int_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(item) for item in value]
    return [int(value)]


def _even_split(total, parts):
    base = total // parts
    remainder = total % parts
    return [base + (1 if i < remainder else 0) for i in range(parts)]


def configure_data_streams(
    total_streams=DEFAULT_M_TOTAL,
    total_resources=DEFAULT_M_TOTAL_RESOURCES,
    num_agents=DEFAULT_AGENT_NUM,
    stream_list=None,
    resource_list=None,
):
    global M_total, m_total, agent_num, M, m, M_max, region_offsets, region_slices

    parsed_streams = _parse_int_list(stream_list)
    parsed_resources = _parse_int_list(resource_list)

    if parsed_streams is None:
        stream_list = _even_split(total_streams, num_agents)
    else:
        stream_list = parsed_streams
    if parsed_resources is None:
        resource_list = _even_split(total_resources, num_agents)
    else:
        resource_list = parsed_resources

    if len(stream_list) != num_agents:
        raise ValueError("stream_list must match agent_num length.")
    if len(resource_list) != num_agents:
        raise ValueError("resource_list must match agent_num length.")
    if sum(stream_list) != total_streams:
        raise ValueError("Sum of stream_list must equal M_total.")
    if sum(resource_list) != total_resources:
        raise ValueError("Sum of resource_list must equal m_total.")

    M_total = sum(stream_list)
    m_total = sum(resource_list)
    agent_num = num_agents
    M = stream_list
    m = resource_list
    M_max = max(stream_list) if stream_list else 0
    region_offsets = np.cumsum([0] + stream_list).tolist()
    region_slices = [
        slice(region_offsets[i], region_offsets[i + 1]) for i in range(num_agents)
    ]


def configure_from_args(args):
    global use_torch_stats, delta, data_configs_delta, eval_data_mode, case_data_path
    configure_data_streams(
        total_streams=getattr(args, "M_total", DEFAULT_M_TOTAL),
        total_resources=getattr(args, "m_total", DEFAULT_M_TOTAL_RESOURCES),
        num_agents=getattr(args, "agent_num", DEFAULT_AGENT_NUM),
        stream_list=getattr(args, "M_list", None),
        resource_list=getattr(args, "m_list", None),
    )
    delta = float(getattr(args, "delta", delta))
    data_configs_delta = delta
    use_torch_stats = getattr(args, "use_torch_stats", False)
    eval_data_mode = getattr(args, "eval_data_mode", getattr(args, "data_mode", "synthetic"))
    case_data_path = getattr(args, "case_data_path", "datasets/IoT_case.pkl")
    # print("Configured data streams:", M)
    _refresh_reference_data()


def sample_multinomial(probs):
    if use_torch_stats:
        try:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            probs_tensor = torch.as_tensor(probs, device=device, dtype=torch.float32)
            indices = torch.multinomial(probs_tensor, num_samples=1).squeeze(1)
            y = torch.zeros_like(probs_tensor)
            y.scatter_(1, indices.unsqueeze(1), 1.0)
            return y.cpu().numpy()
        except Exception:
            pass
    rand = np.random.rand(probs.shape[0], 1)
    cdf = np.cumsum(probs, axis=1)
    indices = (rand > cdf).sum(axis=1)
    y = np.zeros_like(probs)
    y[np.arange(probs.shape[0]), indices] = 1
    return y


def normalize_data(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    normalized_data = (data - mean) / std
    return normalized_data


k = 1950
delta = DEFAULT_DELTA
data_configs_delta = DEFAULT_DELTA
time_range_start = 100
time_range_end = 200


def get_delta():
    return float(delta)


def _refresh_reference_data(sample_size=2000):
    """In-control reference data used by qd/statistics utilities."""
    global train_in_data
    col_blocks = [M_total // 5] * 5
    col_blocks[-1] += M_total - sum(col_blocks)

    cols_normal, cols_t, cols_exp, cols_log, cols_poi = col_blocks
    train_in_data = np.zeros((sample_size, M_total))

    if eval_data_mode == "eval_tep":
        from envs.env_utils.tep_data import load_standardized_tep_arrays

        tep_train, _, _, _ = load_standardized_tep_arrays(case_data_path)
        train_in_data = tep_train.astype(np.float32, copy=False)
        return

    if eval_data_mode == "eval_pi":
        # print("Loading IoT case data for evaluation...")
        import pickle
        with open(case_data_path, 'rb') as f:
            data_dict = pickle.load(f)
        data_ic = np.asarray(data_dict['data_ic'])
        if data_ic.shape[0] != M_total:
            data_ic = data_ic.T
        data_ic = data_ic[:M_total, :]
        test_in_data = data_ic.T
        data = np.zeros((M_total, sample_size))
        for monitor in range(M_total):
            monitor_data = normalize_data(test_in_data[:, monitor])
            rng = np.random.default_rng()
            data[monitor, :] = rng.choice(monitor_data, sample_size, replace=True)
        train_in_data = data.T
        return
    start = 0

    if cols_normal > 0:
        train_in_data[:, start:start + cols_normal] = np.random.normal(size=(sample_size, cols_normal))
        start += cols_normal
    if cols_t > 0:
        train_in_data[:, start:start + cols_t] = normalize_data(
            np.random.standard_t(df=3, size=(sample_size, cols_t))
        )
        start += cols_t
    if cols_exp > 0:
        train_in_data[:, start:start + cols_exp] = normalize_data(
            np.random.exponential(scale=1, size=(sample_size, cols_exp))
        )
        start += cols_exp
    if cols_log > 0:
        train_in_data[:, start:start + cols_log] = normalize_data(
            (np.random.lognormal(mean=0, sigma=0.5, size=(sample_size, cols_log)) - cols_log) / 1.6
        )
        start += cols_log
    if cols_poi > 0:
        train_in_data[:, start:start + cols_poi] = normalize_data(
            (np.random.poisson(lam=10, size=(sample_size, cols_poi)) - 10) / np.sqrt(10)
        )


# configure_data_streams()
# _refresh_reference_data()


eval_data_mode = "synthetic"
case_data_path = "datasets/IoT_case.pkl"
