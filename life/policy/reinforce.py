import torch
from torch import nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return F.softmax(self.fc2(x), dim=1)


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device, net=PolicyNet):
        self.policy_net = net(state_dim, hidden_dim, action_dim).to(device=device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        """根据动作概率分布随机采样"""
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)  # 动作概率分布
        action_dist = torch.distributions.Categorical(probs=probs)  # 创建分类分布
        action = action_dist.sample()  # 从创建的分布中采样
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)

            log_prob = torch.log(self.policy_net(state).gather(1, action))  # log \pi(a|s)
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()
        self.optimizer.step()
