import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ..utils.calculator import compute_advantage


# from .ac.ac import PolicyNet,ValueNet


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)


class PPO:
    """PPO 算法，采用截断的方式"""

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, policy_net=PolicyNet, value_net=ValueNet):
        """
        lmbda:广义优势估计的lambda因子
        epochs: 一条序列的数据用来训练的轮数
        eps: PPO中阶段范围的参数
        """
        self.actor = policy_net(state_dim, hidden_dim, action_dim).to(device)
        self.critic = value_net(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)  # 输出动作的概率分布
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        # 数据类型转换
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)  # 时序差分目标
        td_delta = td_target - self.critic(states)  # 时序差分误差

        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()  # 旧策略

        # 对于actor每采样的一组数据，更新epoch次网络
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)  # 比值

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 对比值进行裁剪

            # 计算loss
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # 对演员的loss，使用ppo目标函数
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            # 优化
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
