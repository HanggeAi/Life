import numpy as np


class Sarsa:
    def __init__(self, n_state, epsilon, alpha, gamma, n_action=4):
        """Sarsa算法

        Arguments:
            ncol -- 环境列数
            nrow -- 环境行数
            epsilon -- 随机选择动作的概率
            alpha -- 学习率
            gamma -- 折扣因子

        Keyword Arguments:
            n_action -- 动作的个数 (default: {4})
        """
        self.Q_table = np.zeros((n_state, n_action))
        self.n_action = n_action
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def take_action(self, state):
        """根据state选择下一步的操作,具体实现为epsilon-贪心"""
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        """用于打印策略"""
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]

        # 若两个动作的价值一样，都会被记录下来
        for i in range(self.n_action):
            if self.Q_table[state][i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        """"更新Q表格"""
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]  # 时序差分误差
        self.Q_table[s0, a0] += self.alpha * td_error


class MultiSarsa:
    """n步Sarsa算法"""

    def __init__(self, n, n_state, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros((n_state, n_action))
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n  # 采用n步Sarsa算法
        self.state_list = []  # 保存之前的状态
        self.action_list = []  # 保存之前的动作
        self.reward_list = []  # 保存之前的奖励

    def take_action(self, state):
        """根据状态图选取一个动作"""
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        """用于输出state下的最优动作(训练完成后)"""
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1, done):
        """基于Sarsa算法,更新Q表格"""
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)

        if len(self.state_list) == self.n:  # 若保存的数据可以进行n步更新
            G = self.Q_table[s1, a1]  # 得到Q(s_{t+n},a_{t+n})
            for i in reversed(range(self.n)):  # 不断向前计算每一步的回报,并折扣累加
                G = self.gamma * G + self.reward_list[i]
                if done and i > 0:  # 虽然最后几步没有到达n步，但是到达了终止状态，也将其更新
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            s = self.state_list.pop(0)  # s_t
            a = self.action_list.pop(0)  # a_t
            self.reward_list.pop(0)  # r_t
            # n步Sarsa的主要更新步骤
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
        if done:
            # 到达终止状态，即将开始下一个序列，将列表清空
            self.state_list.clear()
            self.action_list.clear()
            self.reward_list.clear()
