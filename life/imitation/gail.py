import torch
from torch import nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """判别器模型"""

    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))  # 输出的是一个概率标量


class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d, device, discriminator=Discriminator):
        self.dicriminator = discriminator(state_dim, hidden_dim, action_dim).to(device)
        self.dicriminator_optimizer = torch.optim.Adam(self.dicriminator.parameters(), lr=lr_d)
        self.agent = agent
        self.device = device

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):
        expert_states = torch.tensor(expert_s, dtype=torch.float).to(self.device)
        expert_actions = torch.tensor(expert_a).to(self.device)
        agent_states = torch.tensor(agent_s, dtype=torch.float).to(self.device)
        agent_actions = torch.tensor(agent_a).to(self.device)

        expert_actions = F.one_hot(expert_actions.to(torch.int64), num_classes=2).float()  # 两个动作
        agent_actions = F.one_hot(agent_actions.to(torch.int64), num_classes=2).float()

        expert_prob = self.dicriminator(expert_states, expert_actions)  # 前向传播，输出数据来自于专家的概率
        agent_prob = self.dicriminator(agent_states, agent_actions)
        # 计算判别器的损失
        discriminator_loss = nn.BCELoss()(agent_prob, torch.ones_like(agent_prob)) + \
                             nn.BCELoss()(expert_prob, torch.zeros_like(expert_prob))
        # 优化更新
        self.dicriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.dicriminator_optimizer.step()

        # 将判别器的输出转换为策略的奖励信号
        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,  # 只有rewards改变了，换成了 概率（被判别器识破的概率）
            'next_states': next_s,
            'dones': dones
        }
        self.agent.update(transition_dict)
