import numpy as np
from tqdm import tqdm


def test_agent(agent, env, n_episode):
    """
    对智能体进行episode次测试,记录每个回合的reward,返回其平均值
    """
    return_list = []
    for episode in range(n_episode):
        episode_return = 0
        state = env.reset()
        done = False

        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
    return np.mean(return_list)


def train_bc(bc_agent, env, expert_s, expert_a, n_iterations, batch_size, return_agent=False):
    """训练bc算法的函数"""
    test_returns = []

    with tqdm(total=n_iterations, desc="进度条") as pbar:
        for i in range(n_iterations):
            sample_indices = np.random.randint(0, expert_s.shape[0], size=batch_size)
            expert_s_sample_batch = expert_s[sample_indices]  # 含有重复数据，如本例是从30条经验数据中采样64个
            expert_a_sample_batch = expert_a[sample_indices]

            bc_agent.learn(expert_s_sample_batch, expert_a_sample_batch)  # 有监督的智能体学习

            current_return = test_agent(bc_agent, env, 5)
            test_returns.append(current_return)
            if (i + 1) % 10 == 0:
                pbar.set_postfix({"return": "%.3f" % np.mean(test_returns[-10:])})
            pbar.update(1)
    if return_agent:
        return test_returns, bc_agent
    return test_returns


def train_gail(agent, gail, env, expert_s, expert_a, n_episode=500, return_agent=False):
    """
    gail算法的训练函数
    :param agent: 需要与环境交互的智能体，同时也是要传入gail算法类的智能体
    :param gail: GAIL算法类
    :param env:
    :param expert_s: 专家数据(s,a)中的s
    :param expert_a: 专家数据(s,a)中的a
    :param n_episode:
    :param return_agent:
    :return:
    """
    return_list = []

    with tqdm(total=n_episode, desc="进度条") as pbar:
        for i in range(n_episode):
            episode_return = 0
            state = env.reset()
            done = False
            state_list = []
            action_list = []
            next_state_list = []
            done_list = []

            while not done:
                action = agent.take_action(state)  # 也可换成gail.agent
                next_state, reward, done, _ = env.step(action)
                state_list.append(state)
                action_list.append(action)
                next_state_list.append(next_state)
                done_list.append(done)
                episode_return += reward
                state = next_state
            return_list.append(episode_return)

            gail.learn(expert_s, expert_a,  # 之前的那30条专家数据
                       state_list, action_list, next_state_list, done_list)

            if (i + 1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)
    if return_agent:
        return return_list, gail.agent
    return return_list
