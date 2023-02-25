from life.base.q_learning import QLearning
from life.base.trainer import train_qlearning
from life.envs.cliffwalking import CliffWalkingEnv

agent = QLearning(12 * 4, 0.1, 0.1, 0.9)
env = CliffWalkingEnv(12, 4)
result = train_qlearning(env, agent, )

print(result)
