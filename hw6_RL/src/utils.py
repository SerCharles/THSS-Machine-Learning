import time
import numpy as np
import argparse

def init_args():
    '''
    描述：生成全局参数
    参数：无
    返回：全局参数
    '''
    parser = argparse.ArgumentParser(description = "Arguments of the project.")
    parser.add_argument("--algorithm", type = str, default = 'QLearning')
    parser.add_argument("--num_train", type = int, default = 5000)
    parser.add_argument("--num_test", type = int, default = 1000)

    parser.add_argument("--gamma_q", type = float, default = 0.2)
    parser.add_argument("--lr_q", type = float, default = 1)
    parser.add_argument("--e_q", type = float, default = 1)
    parser.add_argument("--decay_rate_q", type = float, default = 0.99)

    parser.add_argument("--gamma_sarsa", type = float, default = 0.8)
    parser.add_argument("--lr_sarsa", type = float, default = 0.2)
    parser.add_argument("--e_sarsa", type = float, default = 0.5)
    parser.add_argument("--decay_rate_sarsa", type = float, default = 0.4)

    parser.add_argument("--gamma_l", type = float, default = 0.95)
    parser.add_argument("--lr_l", type = float, default = 0.1)
    parser.add_argument("--e_l", type = float, default = 1)
    parser.add_argument("--decay_rate_l", type = float, default = 0.99)
    parser.add_argument("--l", type = float, default = 0.5)
    parser.add_argument("--seed", type = int, default = 1453)
    parser.add_argument("--show", type = int, default = 0)
    args = parser.parse_args()
    return args

def render_single_Q(args, env, Q):
    """Renders Q function once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
    """
    np.random.seed(args.seed)
    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.2)  # Seconds between frames. Modify as you wish.
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward
    print("Episode reward: %f" % episode_reward)


def evaluate_Q(args, env, Q, num_episodes = 100):
    '''
    描述：测试函数
    参数：全局参数，环境，策略函数Q，多少个episode
    返回：平均V
    '''
    np.random.seed(args.seed)
    tot_reward = 0
    for i in range(num_episodes):
        episode_reward = 0
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        tot_reward += episode_reward
    if args.show:
        print("Total", tot_reward, "reward in", num_episodes, "episodes")
        print("Average Reward:", tot_reward / num_episodes)
    return tot_reward / num_episodes
