import numpy as np


def normalize_state(state):
    """
    Normalize the state (dx, dy, vy) to a range suitable for neural network input.
    :param state: (dx, dy, vel_y)
    :return: normalized state as np.array([dx_norm, dy_norm, vy_norm])
    """
    dx, dy, vy, vel_x = state

    dx_norm = np.clip(dx / 212.0, 0, 2)
    dy_norm = np.clip(dy / 200.0, -1.5, 1.5)
    vy_norm = np.clip(vy / 10.0, -1, 1)

    return np.array([dx_norm, dy_norm, vy_norm, vel_x], dtype=np.float32)
