def dis2con(discrete_action, env, action_dim):
    """离散动作 转为 连续动作的函数(将[0,1,2,..,10]映射到[-2,-1.6,...,1.6,2])"""
    action_low = env.action_space.low[0]  # 连续动作的最小值
    action_up = env.action_space.high[0]  # 连续动作的最大值
    out = action_low + (discrete_action / (action_dim - 1)) * (action_up - action_low)
    return out
