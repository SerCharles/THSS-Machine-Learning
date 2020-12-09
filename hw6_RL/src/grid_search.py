from utils import *
from algorithms import *
import gym
import argparse


def grid_search(args):
    '''
    描述：grid search函数
    参数：算法
    返回：最优情况
    '''
    gammas = [0, 0.25, 0.5, 0.8, 0.95, 0.99]
    lrs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1]
    es = [0, 0.25, 0.5, 0.75, 0.9, 1]
    decay_rates = [0, 0.3, 0.6, 0.9, 0.95, 0.99]
    lambdas = [0, 0.25, 0.5, 0.75, 1]

    max_reward = -14530529
    max_dictionary = {'gamma': -1, 'lr': -1, 'e': -1, 'decay_rate': -1, 'lambda': -1, 'train_rewards': [], 'test_reward': max_reward}

    for gamma in gammas:
        for lr in lrs: 
            for e in es: 
                for decay_rate in decay_rates: 
                    for l in lambdas:
                        env = gym.make('Taxi-v3')
                        if args.algorithm == 'Sarsa_lambda':
                            Q, episode_reward = Sarsa_lambda(args, env, args.num_train, gamma = gamma, lr = lr, e = e, decay_rate = decay_rate, l = l)
                        elif args.algorithm == 'Sarsa':
                            if l != 0:
                                continue
                            else:
                                Q, episode_reward = Sarsa(args, env, args.num_train, gamma = gamma, lr = lr, e = e, decay_rate = decay_rate)
                        elif args.algorithm == 'QLearning':
                            if l != 0:
                                continue
                            else:
                                Q, episode_reward = QLearning(args, env, args.num_train, gamma = gamma, lr = lr, e = e, decay_rate = decay_rate)

                        test_reward = evaluate_Q(args, env, Q, args.num_test)
                            
                        print("*" * 100)
                        print("Gamma =", gamma, 'lr =', lr, 'e = ', e, 'decay_rate =', decay_rate, 'lambda =', l)
                        print("Test loss =", test_reward)

                        if test_reward > max_reward:
                            max_reward = test_reward
                            max_dictionary['gamma'] = gamma
                            max_dictionary['lr'] = lr
                            max_dictionary['e'] = e
                            max_dictionary['decay_rate'] = decay_rate
                            max_dictionary['lambda'] = l
                            max_dictionary['train_rewards'] = episode_reward
                            max_dictionary['test_reward'] = test_reward
    print("Gamma =", max_dictionary['gamma'], 'lr =', max_dictionary['lr'], 'e = ', max_dictionary['e'], 'decay_rate =', max_dictionary['decay_rate'], 'lambda =', max_dictionary['lambda'])
    print("Test loss =", max_dictionary['test_reward'])
    return max_dictionary

if __name__ == '__main__':
    args = init_args()
    print(args)
    grid_search(args)