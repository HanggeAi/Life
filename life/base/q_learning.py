import numpy as np
import random


class QLearning:
    """Q-Learning算法"""

    def __init__(self, n_state, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros((n_state, n_action))
        self.n_action = n_action
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def take_action(self, state):
        """根据策略Q选取在state下的最有动作action"""
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        """训练完成后选择最优动作"""
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1):
        """更新Q表格"""
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


class DynaQ:
    def __init__(self, n_state, epsilon, alpha, gamma, n_planning, n_action=4):
        self.Q_table = np.zeros((n_state, n_action))
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning  # 每执行一次Q-learning,执行n_planning次Q-planning
        self.model = dict()  # 每次在真实环境中收集到新数据，就加入到字典中（如果之前不存在的话）

    def take_action(self, state):
        """根据状态选取下一步的动作"""
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def q_learning(self, s0, a0, r, s1):
        """使用Q-learning的方法更新Q表格"""
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def update(self, s0, a0, r, s1):
        """Dyna-Q算法的主要部分,更新Q表格
        使用Q-learning更新一次,在使用Q-planning从历史数据中更新n_planning次"""
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = r, s1  # 将新数据加入到model中
        for _ in range(self.n_planning):  # Q-planning循环
            (s, a), (r, s_) = random.choice(list(self.model.items()))  # 随机选择之前的数据
            self.q_learning(s, a, r, s_)
