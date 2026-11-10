import numpy as np
import gym
import multiprocessing as mp
import random
import os
from gym import spaces
from envs.env_utils import data_configs

# The reviewer archive uses the GPU environment. The legacy CPU vector wrapper
# below is retained only for API compatibility, so its environment class is
# loaded lazily and is not required for PAD GPU testing.
try:  # pragma: no cover - legacy path not used by reviewer scripts
    from envs.online_monitoring import Env
except Exception:  # pragma: no cover
    Env = None


def _get_eval_env():
    from envs.eval_monitoring_aggregate import EvalEnv  # pragma: no cover - legacy path
    return EvalEnv


def _resolve_seed(seed_value, default_seed=1):
    if isinstance(seed_value, (list, tuple)):
        return int(seed_value[0]) if len(seed_value) > 0 else int(default_seed)
    return int(default_seed if seed_value is None else seed_value)


class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    def __init__(self, array_of_param_array):
        super().__init__()
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (
                np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


class SubprocVecEnv(object):
    def __init__(self, all_args):
        """
        envs: list of gym environments to run in subprocesses
        """

        self.num_envs = all_args.n_rollout_threads
        self.closed = False

        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.ps = []
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            for worker_id, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
                p = ctx.Process(
                    target=_subproc_vec_env_worker,
                    args=(work_remote, remote, Env, all_args, worker_id),
                )
                p.daemon = True
                p.start()
                work_remote.close()
                self.ps.append(p)
        finally:
            if cuda_visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        class _RemoteEnvProxy(object):
            def __init__(self, vec_env, env_id):
                object.__setattr__(self, "_vec_env", vec_env)
                object.__setattr__(self, "_env_id", env_id)

            def __getattr__(self, key):
                if key in {"_vec_env", "_env_id"}:
                    return object.__getattribute__(self, key)
                return self._vec_env.get_env_attr(self._env_id, key)

            def __setattr__(self, key, value):
                if key in {"_vec_env", "_env_id"}:
                    object.__setattr__(self, key, value)
                else:
                    self._vec_env.set_env_attr(self._env_id, key, value)

        self.env_list = [_RemoteEnvProxy(self, i) for i in range(self.num_envs)]
        self.remotes[0].send(("get_spaces", None))
        self.num_agent, self.signal_obs_dims, self.signal_action_dims = self.remotes[0].recv()

        # # Heterogeneous v.s. homogeneous action spaces
        # if self.env_list[0].action_dims:
        #     # agent_num-dimensional list of action space dimensions
        #     self.differ_action_dims = self.env_list[0].action_dims
        # else:
        #     self.signal_action_dim = self.env_list[0].action_dim

        self.u_range = 1.0  # control range for continuous control
        self.movable = True

        # environment parameters
        # self.discrete_action_space = True
        self.discrete_action_space = True

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = False

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in range(self.num_agent):
            total_action_space = []

            # physical action space
            if self.discrete_action_space:
                # u_action_space = spaces.Discrete(self.signal_action_dim)  # homogeneous action space
                # u_action_space = spaces.Discrete(self.differ_action_dims[agent])  # heterogeneous action spaces
                # u_action_space = MultiDiscrete([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                #                                 [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                #                                 [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
                action_dim = self.signal_action_dims[agent]
                u_action_space = MultiDiscrete([[0, 1]] * action_dim)
                # the first three are orders to its upstream agent, the last is the quotation of the agent itself
            else:
                u_action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(2,),
                                            dtype=np.float32)  # [-1,1]
            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            obs_dim = self.signal_obs_dims[agent]
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,),
                                                     dtype=np.float32))  # [-inf,inf]

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.num_agent)]

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def get_property(self):
        inv = [env.get_inventory() for env in self.env_list]
        demand = [env.get_demand() for env in self.env_list]
        orders = [env.get_orders() for env in self.env_list]
        return inv, demand, orders

    def set_upper_action(self, upper_action):
        for remote in self.remotes:
            remote.send(("set_attr", ("upper_action", upper_action)))
        for remote in self.remotes:
            remote.recv()

    def set_resources(self, allocations):
        self.set_m(allocations)

    def set_m(self, allocations):
        allocations = np.asarray(allocations, dtype=int)
        if allocations.ndim == 1:
            per_env_allocations = [allocations.tolist()] * self.num_envs
        else:
            per_env_allocations = [allocation.tolist() for allocation in allocations]
        for remote, allocation in zip(self.remotes, per_env_allocations):
            remote.send(("set_attr", ("m", allocation)))
        for remote in self.remotes:
            remote.recv()

    def step_with_m(self, allocations, actions):
        allocations = np.asarray(allocations, dtype=int)
        if allocations.ndim == 1:
            per_env_allocations = [allocations.tolist()] * self.num_envs
        else:
            per_env_allocations = [allocation.tolist() for allocation in allocations]
        for remote, allocation, action in zip(self.remotes, per_env_allocations, actions):
            remote.send(("step_with_m", (allocation, action)))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def set_env_attr(self, env_id, key, value):
        self.remotes[env_id].send(("set_attr", (key, value)))
        return self.remotes[env_id].recv()

    def get_env_attr(self, env_id, key):
        self.remotes[env_id].send(("get_attr", key))
        return self.remotes[env_id].recv()

    def reset(self, bm=None, m=None):
        if m is None:
            per_env_m = [None] * self.num_envs
        else:
            m_arr = np.asarray(m, dtype=int)
            if m_arr.ndim == 1:
                per_env_m = [m_arr.tolist()] * self.num_envs
            else:
                per_env_m = [allocation.tolist() for allocation in m_arr]
        for remote, env_m in zip(self.remotes, per_env_m):
            remote.send(("reset", {"bm": bm, "m": env_m}))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs), None

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array"):
        pass


# single env
class DummyVecEnv(object):
    def __init__(self, all_args):
        """
        envs: list of gym environments to run in subprocesses
        """

        EvalEnv = _get_eval_env()
        self.env_list = [EvalEnv() for i in range(1)]
        self.num_envs = all_args.n_rollout_threads

        self.num_agent = self.env_list[0].agent_num
        self.signal_obs_dims = self.env_list[0].obs_dims
        self.signal_action_dims = self.env_list[0].action_dims

        self.u_range = 1.0  # control range for continuous control
        self.movable = True

        # environment parameters
        self.discrete_action_space = True

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = False
        # in this env, force_discrete_action == False��because world do not have discrete_action

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent_num in range(self.num_agent):
            total_action_space = []

            # physical action space
            if self.discrete_action_space:
                # u_action_space = spaces.Discrete(self.signal_action_dim)
                # u_action_space = MultiDiscrete([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                #                                 [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                #                                 [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
                action_dim = self.signal_action_dims[agent_num]
                u_action_space = MultiDiscrete([[0, 1]] * action_dim)
            else:
                u_action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(2,),
                                            dtype=np.float32)  # [-1,1]
            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            obs_dim = self.signal_obs_dims[agent_num]  # 单个智能体的观测维度
            share_obs_dim += obs_dim
            self.observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))  # [-inf,inf]

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.num_agent)]

    def step(self, actions):

        results = [env.step(action) for env, action in zip(self.env_list, actions)]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, bm=1, seed=1, train=True, test=False, test_delta=None, test_case=False):
        obs = [env.reset(bm=bm, seed=seed, train=train, test=test, test_delta=test_delta, test_case=test_case)
               for env in self.env_list]
        return np.stack(obs), None

    def get_eval_bw_res(self):
        res = self.env_list[0].get_eval_bw_res()
        return res

    def get_eval_num(self):
        eval_num = self.env_list[0].get_eval_num()
        return eval_num

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass


class SubprocVecEnvTest(object):
    def __init__(self, all_args):
        """
        envs: list of gym environments to run in subprocesses
        """

        EvalEnv = _get_eval_env()
        self.num_envs = all_args.n_test_rollout_threads

        self.closed = False
        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.ps = []
        for worker_id, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            p = ctx.Process(
                target=_subproc_vec_env_test_worker,
                args=(work_remote, remote, EvalEnv, all_args, worker_id),
            )
            p.daemon = True
            p.start()
            work_remote.close()
            self.ps.append(p)

        class _RemoteEnvProxy(object):
            def __init__(self, vec_env, env_id):
                object.__setattr__(self, "_vec_env", vec_env)
                object.__setattr__(self, "_env_id", env_id)

            def __getattr__(self, key):
                if key in {"_vec_env", "_env_id"}:
                    return object.__getattribute__(self, key)
                return self._vec_env.get_env_attr(self._env_id, key)

            def __setattr__(self, key, value):
                if key in {"_vec_env", "_env_id"}:
                    object.__setattr__(self, key, value)
                else:
                    self._vec_env.set_env_attr(self._env_id, key, value)

        self.env_list = [_RemoteEnvProxy(self, i) for i in range(self.num_envs)]
        self.remotes[0].send(("get_spaces", None))
        self.num_agent, self.signal_obs_dims, self.signal_action_dims = self.remotes[0].recv()

        # # Heterogeneous v.s. homogeneous action spaces
        # if self.env_list[0].action_dims:
        #     # agent_num-dimensional list of action space dimensions
        #     self.differ_action_dims = self.env_list[0].action_dims
        # else:
        #     self.signal_action_dim = self.env_list[0].action_dim

        self.u_range = 1.0  # control range for continuous control
        self.movable = True

        # environment parameters
        # self.discrete_action_space = True
        self.discrete_action_space = True

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = False

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in range(self.num_agent):
            total_action_space = []

            # physical action space
            if self.discrete_action_space:
                # u_action_space = spaces.Discrete(self.signal_action_dim)  # homogeneous action space
                # u_action_space = spaces.Discrete(self.differ_action_dims[agent])  # heterogeneous action spaces
                # u_action_space = MultiDiscrete([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                #                                 [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
                #                                 [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
                action_dim = self.signal_action_dims[agent]
                u_action_space = MultiDiscrete([[0, 1]] * action_dim)
                # the first three are orders to its upstream agent, the last is the quotation of the agent itself
            else:
                u_action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(2,),
                                            dtype=np.float32)  # [-1,1]
            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            obs_dim = self.signal_obs_dims[agent]
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,),
                                                     dtype=np.float32))  # [-inf,inf]

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.num_agent)]

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step_with_m(self, allocations, actions):
        for remote, allocation, action in zip(self.remotes, allocations, actions):
            remote.send(("step_with_m", (list(allocation), action)))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def get_property(self):
        inv = [env.get_inventory() for env in self.env_list]
        demand = [env.get_demand() for env in self.env_list]
        orders = [env.get_orders() for env in self.env_list]
        return inv, demand, orders

    def set_upper_action(self, upper_action):
        for remote in self.remotes:
            remote.send(("set_attr", ("upper_action", upper_action)))
        for remote in self.remotes:
            remote.recv()

    def set_m(self, allocations):
        for remote, allocation in zip(self.remotes, allocations):
            remote.send(("set_attr", ("m", list(allocation))))
        for remote in self.remotes:
            remote.recv()

    def set_env_attr(self, env_id, key, value):
        self.remotes[env_id].send(("set_attr", (key, value)))
        return self.remotes[env_id].recv()

    def get_env_attr(self, env_id, key):
        self.remotes[env_id].send(("get_attr", key))
        return self.remotes[env_id].recv()

    def reset(self, bm=None, seed=None, train=True, test=False, test_delta=None, test_case=False):
        kwargs = {"bm": bm, "train": train, "test": test, "test_delta": test_delta, "test_case": test_case}
        if seed is not None:
            kwargs["seed"] = seed
        for remote in self.remotes:
            remote.send(("reset", kwargs))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs), None

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array"):
        pass


def _subproc_vec_env_worker(remote, parent_remote, env_fn, all_args, worker_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parent_remote.close()
    data_configs.configure_from_args(all_args)

    worker_seed = _resolve_seed(getattr(all_args, "seed", 1), default_seed=1) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    env = env_fn()

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(env.step(data))
        elif cmd == "step_with_m":
            allocation, action = data
            env.m = list(allocation)
            remote.send(env.step(action))
        elif cmd == "reset":
            data = data or {}
            remote.send(env.reset(**data))
        elif cmd == "get_spaces":
            remote.send((env.agent_num, env.obs_dims, env.action_dims))
        elif cmd == "set_attr":
            key, value = data
            setattr(env, key, value)
            remote.send(True)
        elif cmd == "get_attr":
            remote.send(getattr(env, data))
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError(cmd)


def _subproc_vec_env_test_worker(remote, parent_remote, env_fn, all_args, worker_id):
    parent_remote.close()
    worker_seed = _resolve_seed(getattr(all_args, "seed", 1), default_seed=1) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    data_configs.configure_from_args(all_args)
    env = env_fn()

    def _compact_binary_to_one_hot(compact_actions):
        one_hot_actions = []
        for compact_action in compact_actions:
            compact_action = np.asarray(compact_action, dtype=np.int64).reshape(-1)
            one_hot = np.eye(2, dtype=np.float32)[compact_action].reshape(-1)
            one_hot_actions.append(one_hot)
        return one_hot_actions

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(env.step(data))
        elif cmd == "step_with_m":
            allocation, compact_actions = data
            env.m = list(allocation)
            one_hot_actions = _compact_binary_to_one_hot(compact_actions)
            remote.send(env.step(one_hot_actions))
        elif cmd == "reset":
            data = data or {}
            remote.send(env.reset(**data))
        elif cmd == "get_spaces":
            remote.send((env.agent_num, env.obs_dims, env.action_dims))
        elif cmd == "set_attr":
            key, value = data
            setattr(env, key, value)
            remote.send(True)
        elif cmd == "get_attr":
            remote.send(getattr(env, data))
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError(cmd)
