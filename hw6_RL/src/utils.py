import time
import numpy as np



def render_single_Q(env, Q):
    """Renders Q function once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
    """

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


def evaluate_Q(env, Q, num_episodes=100):
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
    print("Total", tot_reward, "reward in", num_episodes, "episodes")
    print("Average Reward:", tot_reward / num_episodes)
