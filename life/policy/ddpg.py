import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class TwoLayerFC(nn.Module):
    def __init__(self, num_in, num_out, hidden_dim, activation=F.relu, out_fn=lambda x: x) -> None:
        super().__init__()
        self.fc1 = nn.Linear(num_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_out)
        self.activation = activation
        self.out_fn = out_fn

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.out_fn(self.fc3(x))
        return x


class DDPG:
    def __init__(self, num_in_actor, num_out_actor, num_in_critic, hidden_dim,
                 discrete, action_bound, sigma, actor_lr, critic_lr,
                 tau, gamma, device, common_net=TwoLayerFC):
        """
        第一行是神经网络结构上的超参数
        discrete:是否用于处理离散动作
        action_bound:限制动作取值范围
        sigma:用于添加高斯噪声的高斯分布参数
        tau:软更新目标网络的参数
        gamma:衰减因子
        """
        out_fn = (lambda x: x) if discrete else (
            lambda x: torch.tanh(x) * action_bound)
        self.actor = common_net(num_in_actor, num_out_actor, hidden_dim,
                                activation=F.relu, out_fn=out_fn).to(device)
        self.target_actor = common_net(num_in_actor, num_out_actor, hidden_dim,
                                       activation=F.relu, out_fn=out_fn).to(device)
        self.critic = common_net(num_in_critic, 1, hidden_dim).to(device)
        self.target_critic = common_net(
            num_in_critic, 1, hidden_dim).to(device)

        # 设置目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网略并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.action_bound = action_bound
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = num_out_actor
        self.device = device

    def take_action(self, state):
        """输入状态,输出带有噪声的动作"""
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        # 给动作添加噪声，增加探索
        action = action + self.gamma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(
            transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(
            transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(
            transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(
            transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(
            transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 计算critic loss
        next_q_values = self.target_critic(torch.cat([next_states,
                                                      self.target_actor(next_states)],
                                                     dim=1))  # Q_{w-}
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(
            self.critic(torch.cat([states, actions], dim=1)),
            q_targets
        ))
        # 优化
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 计算actor loss
        actor_loss = - \
            torch.mean(self.critic(
                torch.cat([states, self.actor(states)], dim=1)))
        # 优化
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新两个两个目标网络
        self.soft_update(self.critic, self.target_critic)
        self.soft_update(self.actor, self.target_actor)
