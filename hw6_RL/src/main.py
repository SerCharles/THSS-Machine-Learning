import gym
import matplotlib.pyplot as plt
from algorithms import *
from utils import *
import argparse

def main(args):
    '''
    描述：主函数，在最优超参数下训练-测试-绘图
    参数：全局参数
    返回：无
    '''
    env = gym.make('Taxi-v3')

    if args.algorithm == 'QLearning':
        Q1, Q_rewards = QLearning(args, env, args.num_train)
    elif args.algorithm == 'Sarsa':
        Q1, Q_rewards = Sarsa(args, env, args.num_train)
    elif args.algorithm == 'Sarsa_lambda':
        Q1, Q_rewards = Sarsa_lambda(args, env, args.num_train)

    render_single_Q(args, env, Q1)
    evaluate_Q(args, env, Q1, args.num_test)

    plt.plot(range(args.num_train), Q_rewards)
    plt.show()


if __name__ == '__main__':
    args = init_args()
    main(args)
