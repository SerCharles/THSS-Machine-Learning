import numpy as np
import sys


def QLearning(args, env, num_episodes, gamma = 0.95, lr = 0.1, e = 1, decay_rate = 0.99):
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
    np.random.seed(args.seed)

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
            s_next = nexts
            Q[s][a] += lr * (reward + gamma * np.max(Q[s_next]) - Q[s][a])
            tmp_episode_reward += reward
            s = nexts
            if done:
                break
        episode_reward[i] = tmp_episode_reward
        if args.show:
            print("Total reward until episode", i + 1, ":", tmp_episode_reward)
            sys.stdout.flush()
        if i % 10 == 0:
            e = e * decay_rate
    return Q, episode_reward


def Sarsa_lambda(args, env, num_episodes, gamma = 0.95, lr = 0.1, e = 1, decay_rate = 0.99, l = 0.5):
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
    np.random.seed(args.seed)
    Q = np.zeros((env.nS, env.nA))
    E = np.zeros((env.nS, env.nA))

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
            s_next = nexts
            
            if np.random.rand() > e:
                a_next = np.argmax(Q[s_next])
            else:
                a_next = np.random.randint(env.nA)
            delta = reward + gamma * Q[s_next][a_next] - Q[s][a]
            E[s][a] = E[s][a] + 1
            for ss in range(env.nS):
                for aa in range(env.nA):
                    Q[ss][aa] += lr * delta * E[ss][aa]
                    E[ss][aa] = gamma * l * E[ss][aa]
            tmp_episode_reward += reward
            s = s_next
            a = a_next
            if done:
                break
        episode_reward[i] = tmp_episode_reward
        if args.show:
            print("Total reward until episode", i + 1, ":", tmp_episode_reward)
            sys.stdout.flush()
        if i % 10 == 0:
            e = e * decay_rate
    return Q, episode_reward


def Sarsa(args, env, num_episodes, gamma = 0.95, lr = 0.1, e = 1, decay_rate = 0.99):
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
    np.random.seed(args.seed)
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
            s_next = nexts
            
            if np.random.rand() > e:
                a_next = np.argmax(Q[s_next])
            else:
                a_next = np.random.randint(env.nA)

            Q[s][a] += lr * (reward + gamma * Q[s_next][a_next] - Q[s][a])
            tmp_episode_reward += reward
            s = s_next
            a = a_next
            if done:
                break
        episode_reward[i] = tmp_episode_reward
        if args.show:
            print("Total reward until episode", i + 1, ":", tmp_episode_reward)
            sys.stdout.flush()
        if i % 10 == 0:
            e = e * decay_rate
    return Q, episode_reward

