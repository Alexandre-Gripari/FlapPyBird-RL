import numpy as np

from .utils import soft_discretize, ACTIONS, state_action_counts, TOTAL_STATES, update_q, get_q, Q
from ..flappy_env import FlappyEnv

EPISODES = 50000
ALPHA = 1
EPSILON_START = 1
EPSILON_END = 0.005

env = FlappyEnv(render=True, speed=0)

best_score = 0

print(f"Total states: {TOTAL_STATES:,}")

for ep in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        epsilon = max(EPSILON_END,
                      EPSILON_START - (EPSILON_START - EPSILON_END) * (ep / EPISODES))

        # epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * ((1 - ep / EPISODES) ** 1.2)

        dx_bin, dy_bin, vy_bin = soft_discretize(state)

        if np.random.rand() < epsilon:
            action = np.random.randint(0, len(ACTIONS))
        else:
            q_values = get_q(state, Q)
            action = np.argmax(q_values)

        next_state, reward, done, info = env.step(action)
        steps += 1
        total_reward += reward

        update_q(state, action, reward, next_state)

        state = next_state

    if ep % 50 == 0:
        visited_states = np.count_nonzero(np.any(state_action_counts > 0, axis=3))
        explored_percent = visited_states / TOTAL_STATES * 100

        print(f"Ep {ep:5d} | reward: {total_reward:7.1f} | steps: {steps:4d} | score: {info[0]:3d} | "
              f"Visited: {visited_states:5d} ({explored_percent:5.2f}%) | "
              f"ε: {epsilon:.4f}")

np.save("q_matrix_linear_005.npy", Q)

print(f"Best survival: {best_score} steps")
print(f"States visited: {np.sum(np.any(state_action_counts > 0, axis=3)):,} / {TOTAL_STATES:,}")
print(f"State-action pairs explored: {np.sum(state_action_counts > 0):,}")
