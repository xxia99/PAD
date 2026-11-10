# import matplotlib
# import matplotlib.pyplot as plt
import itertools
from math import comb
import numpy as np
from envs.env_utils import data_configs


def init_state(total_streams, m=None):
    state = np.zeros((2, total_streams))
    # state = [0] * 2 * M
#     non_ob = random.sample(list(np.arange(0, M)), M - m)
#     state[1, non_ob] = -1
    return state


def init_action(total_streams, resources):
    action_dict = {}
    all_action = list(itertools.combinations(range(total_streams), resources))
    for i, action in enumerate(all_action):
        temp = np.zeros((total_streams,))
        temp[list(action)] = 1
        action_dict[i] = temp
    return action_dict


def action_sim(a, b, resources):
    dif = np.array(a) - np.array(b)
    sim = np.sum(dif * np.transpose(dif)) / (2 * resources)
    return sim


# given the observed data of a single data stream, calculate the cumulative intervals (positive)
# qd is learned and fixed using in-control training data
def CI_p(monitor_data, bins):
    sorted_data = np.sort(monitor_data)
    length = len(sorted_data)
    interval_length = length / bins
    qd = np.zeros(bins - 1)
    for i in np.arange(1, bins, 1):
        qd[i - 1] = sorted_data[int(i * interval_length)]
    return qd


def all_qd(total_streams, train_data):
    qd_total = np.zeros((data_configs.d - 1, total_streams))
    for monitor in range(total_streams):
        train_in_data_monitor = train_data[:, monitor]
        qd_total[:, monitor] = CI_p(train_in_data_monitor, data_configs.d)
    return qd_total


# given qd, which interval Y the new data belongs to
def Y(t_data, qd, bins):
    index = np.searchsorted(qd, t_data, side="left")
    result = np.zeros(bins)
    result[index] = 1
    return result


def A(result, bins):
    An = np.zeros(bins - 1)
    for i in range(len(An)):
        An[i] = np.sum(result[0:i + 1])
    Ap = 1 - An
    return Ap, An


def local_stat_calp(t_data, Ap, allowance, loc_stat_prep):  # each monitor has local_stat dimension 2x(d-1)
    S1_pre = loc_stat_prep[:, 0]
    S2_pre = loc_stat_prep[:, 1]
    loc_stat_newp = np.zeros((data_configs.d - 1, 2))
    g = 1 - np.arange(1, data_configs.d) / data_configs.d
    C = (S1_pre - S2_pre + Ap - g).T @ np.linalg.inv(np.diag(S2_pre + g)) @ (S1_pre - S2_pre + Ap - g)
    Wp = max(0, C - allowance)
    if C > allowance:
        loc_stat_newp[:, 0] = (S1_pre + Ap) * (C - allowance) / C
        loc_stat_newp[:, 1] = (S2_pre + g) * (C - allowance) / C
    else:
        loc_stat_newp[:, 0] = np.zeros(data_configs.d - 1)
        loc_stat_newp[:, 1] = np.zeros(data_configs.d - 1)
    return Wp, loc_stat_newp


def local_stat_caln(t_data, An, allowance, loc_stat_pren):  # each monitor has local_stat dimension 2x(d-1)
    S1_pre = loc_stat_pren[:, 0]
    S2_pre = loc_stat_pren[:, 1]
    loc_stat_newn = np.zeros((data_configs.d - 1, 2))
    g = np.arange(1, data_configs.d) / data_configs.d
    C = (S1_pre - S2_pre + An - g).T @ np.linalg.inv(np.diag(S2_pre + g)) @ (S1_pre - S2_pre + An - g)
    Wn = max(0, C - allowance)
    if C > allowance:
        loc_stat_newn[:, 0] = (S1_pre + An) * (C - allowance) / C
        loc_stat_newn[:, 1] = (S2_pre + g) * (C - allowance) / C
    else:
        loc_stat_newn[:, 0] = np.zeros(data_configs.d - 1)
        loc_stat_newn[:, 1] = np.zeros(data_configs.d - 1)
    return Wn, loc_stat_newn


# the input A are pesudo observations, so only calculate statistics, but not updating the S
def un_local_stat_calp(t_data, Ap, allowance, loc_stat_prep):
    S1_pre = loc_stat_prep[:, 0]
    S2_pre = loc_stat_prep[:, 1]
    loc_stat_newp = np.zeros((data_configs.d - 1, 2))
    g = 1 - np.arange(1, data_configs.d) / data_configs.d
    C = (S1_pre - S2_pre + Ap - g).T @ np.linalg.inv(np.diag(S2_pre + g)) @ (S1_pre - S2_pre + Ap - g)
    Wp = max(0, C - allowance)
    loc_stat_newp[:, 0] = S1_pre
    loc_stat_newp[:, 1] = S2_pre
    return Wp, loc_stat_newp


def un_local_stat_caln(t_data, An, allowance, loc_stat_pren):
    S1_pre = loc_stat_pren[:, 0]
    S2_pre = loc_stat_pren[:, 1]
    loc_stat_newn = np.zeros((data_configs.d - 1, 2))
    g = np.arange(1, data_configs.d) / data_configs.d
    C = (S1_pre - S2_pre + An - g).T @ np.linalg.inv(np.diag(S2_pre + g)) @ (S1_pre - S2_pre + An - g)
    Wn = max(0, C - allowance)
    loc_stat_newn[:, 0] = S1_pre
    loc_stat_newn[:, 1] = S2_pre
    return Wn, loc_stat_newn


if __name__ == "__main__":
    qd_total = all_qd(data_configs.M_total, data_configs.train_in_data)
    resources = data_configs.m[0] if isinstance(data_configs.m, list) else data_configs.m
    action_length = comb(data_configs.M_total, resources)
    action_dict = init_action(data_configs.M_total, resources)
    print("action_length: ", action_length)
    print("action_dict: ", action_dict)
