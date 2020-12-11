from utils import *
from algorithms import *
import gym
import argparse


def grid_search(args):
    '''
    描述：grid search函数(搜Q/Sarsa的)
    参数：全局参数
    返回：最优情况
    '''
    gammas = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95, 0.99]
    lrs = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 0.2, 0.5, 1]
    es = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 1]
    decay_rates = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]

    max_reward = -14530529
    max_dictionary = {'gamma': -1, 'lr': -1, 'e': -1, 'decay_rate': -1, 'train_rewards': [], 'test_reward': max_reward}

    for gamma in gammas:
        for lr in lrs: 
            for e in es: 
                for decay_rate in decay_rates: 
                    env = gym.make('Taxi-v3')
                    if args.algorithm == 'Sarsa':
                        Q, episode_reward = Sarsa(args, env, args.num_train, gamma = gamma, lr = lr, e = e, decay_rate = decay_rate)
                    elif args.algorithm == 'QLearning':
                        Q, episode_reward = QLearning(args, env, args.num_train, gamma = gamma, lr = lr, e = e, decay_rate = decay_rate)

                    test_reward = evaluate_Q(args, env, Q, args.num_test)
                            
                    print("*" * 100)
                    print("Gamma =", gamma, 'lr =', lr, 'e = ', e, 'decay_rate =', decay_rate)
                    print("Test loss =", test_reward)

                    if test_reward > max_reward:
                        max_reward = test_reward
                        max_dictionary['gamma'] = gamma
                        max_dictionary['lr'] = lr
                        max_dictionary['e'] = e
                        max_dictionary['decay_rate'] = decay_rate
                        max_dictionary['train_rewards'] = episode_reward
                        max_dictionary['test_reward'] = test_reward

    print("Gamma =", max_dictionary['gamma'], 'lr =', max_dictionary['lr'], 'e = ', max_dictionary['e'], 'decay_rate =', max_dictionary['decay_rate'])
    print("Test loss =", max_dictionary['test_reward'])
    return max_dictionary

def grid_search_lambda(args):
    '''
    描述：grid search函数(搜Q/Sarsa的)
    参数：全局参数
    返回：最优情况
    '''
    hyper_types = ['q', 'sarsa', 'lambda']
    lambdas = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1]
    max_reward = -14530529
    max_dictionary = {'gamma': -1, 'lr': -1, 'e': -1, 'decay_rate': -1, 'lambda' : -1, 'train_rewards': [], 'test_reward': max_reward}

    for hyper_type in hyper_types:
        env = gym.make('Taxi-v3')
        if hyper_type == 'q':
            gamma = args.gamma_q
            lr = args.lr_q
            e = args.e_q
            decay_rate = args.decay_rate_q
        elif hyper_type == 'sarsa':
            gamma = args.gamma_sarsa
            lr = args.lr_sarsa
            e = args.e_sarsa
            decay_rate = args.decay_rate_sarsa
        elif hyper_type == 'lambda':
            gamma = args.gamma_l
            lr = args.lr_l
            e = args.e_l
            decay_rate = args.decay_rate_l
        for l in lambdas:
            Q, episode_reward = Sarsa_lambda(args, env, args.num_train, \
                gamma = gamma, lr = lr, e = e, decay_rate = decay_rate, l = l)
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
    if args.algorithm == 'Sarsa_lambda':
        grid_search_lambda(args)
    else: 
        grid_search(args)

