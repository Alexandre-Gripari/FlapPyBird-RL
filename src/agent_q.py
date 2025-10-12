import argparse

import numpy as np

from .flappy_env import FlappyEnv

DX_MIN, DX_MAX = 0, 212
DY_MIN, DY_MAX = -104, 256
VEL_Y_MIN, VEL_Y_MAX = -8, 10

DX_BINS = 24
DY_BINS = 36
VEL_Y_BINS = 5
ACTIONS = [0, 1]


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


def get_Q(state, Q):
    """
    Get Q-values for a given state
    :param state: (dx, dy, vel_y)
    :param Q: Q-matrix
    :return: Q-values for both actions
    """
    dx_bin, dy_bin, vy_bin = soft_discretize(state)
    return Q[dx_bin, dy_bin, vy_bin, :]


def play_game(render=True, speed=30, verbose=True, Q=None):
    """
    Play one game with the trained agent
    :param render: whether to render the game
    :param speed: game speed (higher is faster)
    :param verbose: whether to print game stats
    :param Q: Q-matrix
    :return: (steps, pipes_passed, total_reward)
    where:
        steps: number of steps taken
        pipes_passed: number of pipes passed
        total_reward: total reward accumulated
    """
    env = FlappyEnv(render=render, speed=speed)
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    last_2_pipes = []

    while not done:
        q_values = get_Q(state, Q)
        action = np.argmax(q_values)

        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        last_2_pipes = [info[1].y if info[1] is not None else None, info[2].y if info[2] is not None else None]

    env.close()

    if verbose:
        print(f"   Game finished!")
        print(f"   Last state: dx={state[0]}, dy={state[1]}, vel_y={state[2]}")
        print(f"   Last 2 pipes: {last_2_pipes}")
        print(f"   Steps: {steps}")
        print(f"   Pipes passed: {info[0]}")
        print(f"   Total reward: {total_reward:.1f}")

    return steps, info[0], total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_matrix_path", type=str, default="q_matrix_final.npy")
    args = parser.parse_args()

    Q = np.load(args.q_matrix_path, allow_pickle=True)
    total_learned_states = np.count_nonzero(np.sum(Q, axis=3))
    print(f"Number of learned states: {total_learned_states} / {DX_BINS * DY_BINS * VEL_Y_BINS * len(ACTIONS)}")
    play_game(render=True, speed=30)
