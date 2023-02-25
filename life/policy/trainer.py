from tqdm import tqdm
import numpy as np


def train_reinforce(agent, env, num_episodes, return_agent=False):
    """REINFORCE算法的训练函数"""
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    if return_agent:
        return return_list, agent
    return return_list


def train_ac(agent, env, num_episodes, return_agent=False):
    """ac算法的训练函数"""
    out = train_reinforce(agent, env, num_episodes, return_agent=return_agent)
    return out


def train_ppo(agent, env, num_episodes, return_agent=False):
    """ppo算法的训练函数"""
    out = train_reinforce(agent, env, num_episodes, return_agent=return_agent)
    return out


# #################################################
# 深度确定性策略梯度属于off policy
def train_ddpg(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, return_agent=False):
    """DDPG算法的训练函数"""
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    if return_agent:
        return return_list, agent
    return return_list


def train_sac(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, return_agent=False):
    """训练sac算法的函数"""
    out = train_ddpg(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, return_agent=return_agent)
    return out
