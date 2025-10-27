import argparse

import numpy as np

from .utils import TOTAL_STATES, get_q
from ..flappy_env import FlappyEnv


def play_game(render=True, speed=30, verbose=True, q=None):
    """
    Play one game with the trained agent
    :param render: whether to render the game
    :param speed: game speed (higher is faster)
    :param verbose: whether to print game stats
    :param q: Q-matrix
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
        q_values = get_q(state, q)
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
    parser.add_argument("--benchmark", action="store_true", help="Launch a benchmark of 100 games")
    args = parser.parse_args()

    Q = np.load(args.q_matrix_path, allow_pickle=True)
    total_learned_states = np.count_nonzero(np.sum(Q, axis=3))
    print(f"Number of learned states: {total_learned_states} / {TOTAL_STATES}")

    if args.benchmark:
        scores = []
        for i in range(100):
            _, score, _ = play_game(render=False, speed=0, verbose=False, q=Q)
            scores.append(score)
            if i % 10 == 0:
                print(f"Game {i}")
        print(f"Benchmark sur 100 parties :")
        print(f"  Score moyen : {np.mean(scores):.2f}")
        print(f"  Meilleur score : {np.max(scores)}")
        print(f"  Pire score : {np.min(scores)}")
    else:
        play_game(render=True, speed=30, q=Q)
