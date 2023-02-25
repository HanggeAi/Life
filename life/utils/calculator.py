import torch
import numpy as np


def compute_advantage(gamma, lmbda, td_delta):
    """计算优势函数"""
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)  # 妙啊，边累计计算advantage,边加入列表 以保存每一个时间步的advantage.
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def sample_expert_data(env, agent, n_episodes):
    """
    生成专家的与环境交互的轨迹数据
    env:专家所在的环境
    agent:专家智能体
    n_episode:轨迹个数
    """
    states = []
    actions = []
    for episode in range(n_episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.take_action(state)
            states.append(state)
            actions.append(action)

            next_state, reward, done, _ = env.step(action)
            state = next_state
    return np.array(states), np.array(actions)


def moving_average(a, window_size):
    """数据平滑处理"""
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))