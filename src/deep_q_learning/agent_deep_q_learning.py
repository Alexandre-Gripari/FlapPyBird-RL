import numpy as np
import torch

from src.flappy_env import FlappyEnv
from .q_network import QNetwork

agent = QNetwork(3, 2)
agent.load_state_dict(torch.load("models/flappy_dqn_model.pth"))
agent.eval()

DX_MIN, DX_MAX = 0, 212
DY_MIN, DY_MAX = -104, 256
VEL_Y_MIN, VEL_Y_MAX = -8, 10

DX_BINS = 24
DY_BINS = 36
VEL_Y_BINS = 5
ACTIONS = [0, 1]


def normalize_state(state):
    """
    Normalize the state (dx, dy, vy) to a range suitable for neural network input.
    :param state: (dx, dy, vel_y)
    :return: normalized state as np.array([dx_norm, dy_norm, vy_norm])
    """
    dx, dy, vy = state

    dx_norm = np.clip(dx / 212.0, 0, 2)
    dy_norm = np.clip(dy / 200.0, -1.5, 1.5)
    vy_norm = np.clip(vy / 10.0, -1, 1)

    return np.array([dx_norm, dy_norm, vy_norm], dtype=np.float32)


def play_game(render=True, speed=30, verbose=True):
    """
    Play one game with the trained agent
    :param render: whether to render the game
    :param speed: game speed (higher is faster)
    :param verbose: whether to print game stats
    :return: (steps, pipes_passed, total_reward)
    where:
        steps: number of steps taken
        pipes_passed: number of pipes passed
        total_reward: total reward accumulated
    """
    env = FlappyEnv(render=render, speed=speed)
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
    play_game(render=True, speed=0, verbose=True)
