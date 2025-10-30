import argparse

import numpy as np
import torch

from src.deep_q_learning.utils import normalize_state
from src.flappy_env import FlappyEnv
from .q_network import QNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--increase_difficulty", type=bool, default=False)
parser.add_argument("--benchmark", action="store_true", help="Launch a benchmark of 100 games")
parser.add_argument("--velocity_model", type=bool, default=False,
                    help="Use the model trained with velocity in the state")
args = parser.parse_args()

agent = QNetwork(4 if args.velocity_model else 3, 2)
agent.load_state_dict(
    torch.load("models/flappy_dqn_model_vel.pth" if args.velocity_model else "models/flappy_dqn_model.pth"))
agent.eval()


def play_game(render=True, speed=30, verbose=True, increase_difficulty=False):
    """
    Play one game with the trained agent
    :param render: whether to render the game
    :param speed: game speed (higher is faster)
    :param verbose: whether to print game stats
    :param increase_difficulty: whether to increase difficulty over time (faster pipes)
    :return: (steps, pipes_passed, total_reward)
    where:
        steps: number of steps taken
        pipes_passed: number of pipes passed
        total_reward: total reward accumulated
    """
    env = FlappyEnv(render=render, speed=speed, increase_difficulty=increase_difficulty)
    state = env.reset()
    state = normalize_state(state)
    done = False
    total_reward = 0
    steps = 0
    score = 0

    while not done:
        with torch.no_grad():
            q_values = agent(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).detach().numpy()[0]
        action = np.argmax(q_values)

        state, reward, done, score = env.step(action)
        state = normalize_state(state)
        total_reward += reward
        steps += 1

    env.close()

    if verbose:
        print(f"  Game finished!")
        print(f"   Steps: {steps}")
        print(f"   Pipes passed: {score}")
        print(f"   Total reward: {total_reward:.1f}")

    return steps, score, total_reward


if __name__ == "__main__":
    play_game(render=True, speed=0, verbose=True, increase_difficulty=args.increase_difficulty)

    if args.benchmark:
        scores = []
        for i in range(100):
            _, score, _ = play_game(render=False, speed=0, verbose=False, increase_difficulty=args.increase_difficulty)
            scores.append(score[0])
            if i % 10 == 0:
                print(f"Game {i}")
        print(f"Benchmark sur 100 parties :")
        print(f"  Score moyen : {np.mean(scores):.2f}")
        print(f"  Meilleur score : {np.max(scores)}")
        print(f"  Pire score : {np.min(scores)}")
