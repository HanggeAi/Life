import torch
from torch import nn
import torch.nn.functional as F
from ..reinforce import PolicyNet


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)  # 注意这是一个回归问题


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device,
                 policy_net=PolicyNet, value_net=ValueNet):
        # 定义策略网络 和 价值网络
        self.actor = policy_net(state_dim, hidden_dim, action_dim).to(device)
        self.critic = value_net(state_dim, hidden_dim).to(device)

        # 分别为两个网络建立优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)  # 根据概率大小采样
        action = action_dist.sample()
        return action.item()  # 输出标量

    def update(self, transition_dict):
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
        log_probs = torch.log(self.actor(states).gather(1, actions))

        # 计算两个网络的loss
        actor_loss = torch.mean(-log_probs * td_delta.detach())  # 策略的损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        # 更新网络参数
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # 误差反向传播
        actor_loss.backward()
        critic_loss.backward()

        # 优化器step()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
