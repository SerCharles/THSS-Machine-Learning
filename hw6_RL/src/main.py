import gym
import matplotlib.pyplot as plt
from algorithms import QLearning
from utils import render_single_Q, evaluate_Q


# Feel free to run your own debug code in main!
def main():
    num_episodes = 5000
    env = gym.make('Taxi-v3')

    # q_learning
    Q1, Q_rewards = QLearning(env, num_episodes)
    render_single_Q(env, Q1)
    evaluate_Q(env, Q1, 200)

    plt.plot(range(num_episodes), Q_rewards)
    plt.show()


if __name__ == '__main__':
    main()
