from life.dqn.dqn_improved import DuelingDQN
from life.dqn.trainer import train
from life.utils.replay.replay_buffer import ReplayBuffer
from life.envs.con_env_demo import make
import gym
import torch

lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cpu")
env = make()
state_dim = env.observation_space.shape[0]
action_dim = 11
agent = DuelingDQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                   target_update, device)
replay_buffer = ReplayBuffer(buffer_size)
result = train(agent, env, replay_buffer, minimal_size, batch_size, con_act=True)
print(result)
