import numpy as np

GAMMA = 0.95

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


def update_q(state, action, reward, next_state):
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

    current_q = Q[dx_bin, dy_bin, vy_bin, action]
    max_next_q = np.max(Q[next_dx_bin, next_dy_bin, next_vy_bin, :])

    Q[dx_bin, dy_bin, vy_bin, action] = current_q + alpha * (reward + GAMMA * max_next_q - current_q)

def get_q(state, q):
    """
    Get Q-values for a given state
    :param state: (dx, dy, vel_y)
    :param q: Q-matrix
    :return: Q-values for both actions
    """
    dx_bin, dy_bin, vy_bin = soft_discretize(state)
    return q[dx_bin, dy_bin, vy_bin, :]
