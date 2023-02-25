import torch
from ..policy.ppo import PolicyNet


class BehaviorClone:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, device, policy_net=PolicyNet):
        self.policy = policy_net(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.device = device

    def learn(self, states, actions):
        """policy net 学习，参数更新"""
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1).to(self.device)
        log_probs = torch.log(self.policy(states).gather(1, actions))  # 注意这里的损失函数计算方式
        bc_loss = torch.mean(-log_probs)  # 最大似然估计

        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
