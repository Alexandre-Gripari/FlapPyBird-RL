import numpy as np


def get_learning_rate(dx_bin, dy_bin, vy_bin, action, state_action_counts):
    """
    Calculate adaptive learning rate based on state-action visit count.
    α = 1 / (1 + N(s,a))
    :param dx_bin: discretized dx
    :param dy_bin: discretized dy
    :param vy_bin: discretized vel_y
    :param action: action taken (0 or 1)
    :param state_action_counts : 4D array of state-action visit counts
    :return: learning rate
    """
    count = state_action_counts[dx_bin, dy_bin, vy_bin, action]
    return 1.0 / (1.0 + count)


def soft_discretize(state, config):
    """
    Discretize the state (dx, dy, vy) into bins for tabular Q-learning.
    :param state: (dx, dy, vel_y)
    :param config: TrainingConfig instance
    :return: (dx_bin, dy_bin, vy_bin)
    """
    dx, dy, vy, _ = state

    dx = np.clip(dx, config.dx_min, config.dx_max)
    dy = np.clip(dy, config.dy_min, config.dy_max)
    vy = np.clip(vy, config.vel_y_min, config.vel_y_max)

    dx_bin = int((dx - config.dx_min) / (config.dx_max - config.dx_min) * (config.dx_bins - 1))
    dy_bin = int((dy - config.dy_min) / (config.dy_max - config.dy_min) * (config.dy_bins - 1))
    vy_bin = int((vy - config.vel_y_min) / (config.vel_y_max - config.vel_y_min) * (config.vel_y_bins - 1))

    dx_bin = np.clip(dx_bin, 0, config.dx_bins - 1)
    dy_bin = np.clip(dy_bin, 0, config.dy_bins - 1)
    vy_bin = np.clip(vy_bin, 0, config.vel_y_bins - 1)

    return dx_bin, dy_bin, vy_bin


def get_q(state, q, config):
    """
    Get Q-values for a given state
    :param state: (dx, dy, vel_y)
    :param q: Q-matrix
    :param config: TrainingConfig instance
    :return: Q-values for both actions
    """
    dx_bin, dy_bin, vy_bin = soft_discretize(state, config)
    return q[dx_bin, dy_bin, vy_bin, :]
