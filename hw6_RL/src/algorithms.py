import numpy as np
import sys


def QLearning(env, num_episodes=5000, gamma=0.95, lr=0.1, e=1, decay_rate=0.99):
    # num_episodes=5000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99
    """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
    num_episodes: int
    Number of episodes of training.
    gamma: float
    Discount factor. Number in range [0, 1)
    learning_rate: float
    Learning rate. Number in range [0, 1)
    e: float
    Epsilon value used in the epsilon-greedy method.
    decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    np.array
    An array of shape [env.nS x env.nA] representing state, action values
    """
    Q = np.zeros((env.nS, env.nA))
    episode_reward = np.zeros((num_episodes,))
    for i in range(num_episodes):
        tmp_episode_reward = 0
        s = env.reset()
        while (True):
            if np.random.rand() > e:
                a = np.argmax(Q[s])
            else:
                a = np.random.randint(env.nA)
            nexts, reward, done, info = env.step(a)
            Q[s][a] += lr * (reward + gamma * np.max(Q[nexts]) - Q[s][a])
            tmp_episode_reward += reward
            s = nexts
            if done:
                break
        episode_reward[i] = tmp_episode_reward
        print("Total reward until episode", i + 1, ":", tmp_episode_reward)
        sys.stdout.flush()
        if i % 10 == 0:
            e = e * decay_rate
    return Q, episode_reward


def Sarsa_lambda(env, num_episodes=5000, gamma=0.95, lr=0.1, e=1, decay_rate=0.99, l=0.5):
    # num_episodes=5000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99
    """Learn state-action values using the Sarsa lambda algorithm with epsilon-greedy exploration strategy.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
    num_episodes: int
    Number of episodes of training.
    gamma: float
    Discount factor. Number in range [0, 1)
    learning_rate: float
    Learning rate. Number in range [0, 1)
    e: float
    Epsilon value used in the epsilon-greedy method.
    decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)
    l: float
    weight of TD learning. Number in range [0, 1)

    Returns
    -------
    np.array
    An array of shape [env.nS x env.nA] representing state, action values
    """

    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    pass


def Sarsa(env, num_episodes=5000, gamma=0.95, lr=0.1, e=1, decay_rate=0.99):
    # num_episodes=5000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99
    """Learn state-action values using the Sarsa algorithm with epsilon-greedy exploration strategy.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
    num_episodes: int
    Number of episodes of training.
    gamma: float
    Discount factor. Number in range [0, 1)
    learning_rate: float
    Learning rate. Number in range [0, 1)
    e: float
    Epsilon value used in the epsilon-greedy method.
    decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    np.array
    An array of shape [env.nS x env.nA] representing state, action values
    """

    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    pass
