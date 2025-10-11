import numpy as np

from .flappy_env import FlappyEnv

GAMMA = 0.95
EPISODES = 20000
ALPHA = 1
EPSILON_START = 1
EPSILON_END = 0.05

BIRD_WIDTH = 34
BIRD_HEIGHT = 24

DX_MIN, DX_MAX = 0, 212
DY_MIN, DY_MAX = -104, 256
VEL_Y_MIN, VEL_Y_MAX = -8, 10
ACTIONS = [0, 1]

DX_BINS = 24
DY_BINS = 36
VEL_Y_BINS = 5

TOTAL_STATES = DX_BINS * DY_BINS * VEL_Y_BINS * len(ACTIONS)

Q = np.zeros((DX_BINS, DY_BINS, VEL_Y_BINS, len(ACTIONS)), dtype=np.float32)
state_action_counts = np.zeros((DX_BINS, DY_BINS, VEL_Y_BINS, len(ACTIONS)), dtype=np.int32)

env = FlappyEnv(render=True, speed=1000)


def soft_discretize(state):
    """
    Discretize the state (dx, dy, vy) into bins for tabular Q-learning.
    :param state: (dx, dy, vel_y)
    :return: (dx_bin, dy_bin, vy_bin)
    """
    dx, dy, vy = state

    dx = np.clip(dx, DX_MIN, DX_MAX)
    dy = np.clip(dy, DY_MIN, DY_MAX)
    vy = np.clip(vy, VEL_Y_MIN, VEL_Y_MAX)

    dx_bin = int((dx - DX_MIN) / (DX_MAX - DX_MIN) * (DX_BINS - 1))
    dy_bin = int((dy - DY_MIN) / (DY_MAX - DY_MIN) * (DY_BINS - 1))
    vy_bin = int((vy - VEL_Y_MIN) / (VEL_Y_MAX - VEL_Y_MIN) * (VEL_Y_BINS - 1))

    dx_bin = np.clip(dx_bin, 0, DX_BINS - 1)
    dy_bin = np.clip(dy_bin, 0, DY_BINS - 1)
    vy_bin = np.clip(vy_bin, 0, VEL_Y_BINS - 1)

    return dx_bin, dy_bin, vy_bin


def get_learning_rate(dx_bin, dy_bin, vy_bin, action):
    """
    Calculate adaptive learning rate based on state-action visit count.
    α = 1 / (1 + N(s,a))
    :param dx_bin: discretized dx
    :param dy_bin: discretized dy
    :param vy_bin: discretized vel_y
    :param action: action taken (0 or 1)
    :return: learning rate
    """
    count = state_action_counts[dx_bin, dy_bin, vy_bin, action]
    return 1.0 / (1.0 + count)


def update_Q(state, action, reward, next_state):
    """
    Update Q-value using the Q-learning formula.
    Q(s,a) = Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))
    :param state: current state (dx, dy, vel_y)
    :param action: action taken (0 or 1)
    :param reward: reward received
    :param next_state: next state (dx', dy', vel_y')
    :return: None
    """
    dx_bin, dy_bin, vy_bin = soft_discretize(state)
    next_dx_bin, next_dy_bin, next_vy_bin = soft_discretize(next_state)

    alpha = get_learning_rate(dx_bin, dy_bin, vy_bin, action)
    state_action_counts[dx_bin, dy_bin, vy_bin, action] += 1

    current_Q = Q[dx_bin, dy_bin, vy_bin, action]
    max_next_Q = np.max(Q[next_dx_bin, next_dy_bin, next_vy_bin, :])

    Q[dx_bin, dy_bin, vy_bin, action] = current_Q + alpha * (reward + GAMMA * max_next_Q - current_Q)


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

        dx_bin, dy_bin, vy_bin = soft_discretize(state)

        if np.random.rand() < epsilon:
            action = np.random.randint(0, len(ACTIONS))
        else:
            q_values = Q[dx_bin, dy_bin, vy_bin, :]
            action = np.argmax(q_values)

        next_state, reward, done = env.step(action)
        steps += 1
        total_reward += reward

        update_Q(state, action, reward, next_state)

        state = next_state

    if ep % 50 == 0:
        visited_states = np.count_nonzero(np.any(state_action_counts > 0, axis=3))
        explored_percent = visited_states / TOTAL_STATES * 100

        print(f"Ep {ep:5d} | reward: {total_reward:7.1f} | steps: {steps:4d} | "
              f"Visited: {visited_states:5d} ({explored_percent:5.2f}%) | "
              f"ε: {epsilon:.4f}")

np.save("q_matrix_final_2.npy", Q)

print(f"Best survival: {best_score} steps")
print(f"States visited: {np.sum(np.any(state_action_counts > 0, axis=3)):,} / {TOTAL_STATES:,}")
print(f"State-action pairs explored: {np.sum(state_action_counts > 0):,}")

print(f"\nTesting trained agent (5 games)...\n")

env_test = FlappyEnv(render=False, speed=30)
test_scores = []

for game in range(5):
    state = env_test.reset()
    done = False
    steps = 0

    while not done:
        dx_bin, dy_bin, vy_bin = soft_discretize(state)

        q_values = Q[dx_bin, dy_bin, vy_bin, :]
        action = np.argmax(q_values)

        state, reward, done = env_test.step(action)
        steps += 1

    test_scores.append(steps)
    pipes = steps // 100
    print(f"Game {game + 1}: {steps} steps (~{pipes} pipes)")

env_test.close()

print(f"\n Test Results:")
print(f"   Average: {np.mean(test_scores):.1f} steps")
print(f"   Best: {np.max(test_scores)} steps")
print(f"   Worst: {np.min(test_scores)} steps")
