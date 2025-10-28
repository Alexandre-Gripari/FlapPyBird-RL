import numpy as np
import torch

from src.flappy_env import FlappyEnv
from .dqn_agent import DQNAgent

GAMMA = 0.99
EPISODES = 20000
LEARNING_RATE = 5e-4
EPSILON_START = 1.0
EPSILON_END = 0.0005
EPSILON_DECAY = 0.999975

dqn_agent = DQNAgent(
    input_size=3,
    output_size=2,
    epsilon=EPSILON_START,
    gamma=GAMMA,
    lr=LEARNING_RATE,
    batch_size=256,

)

env = FlappyEnv(render=False, speed=0, deep_qlearning=True)


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


TARGET_UPDATE = 1000
best_score = 0
total_reward = 0
avg_score = 0
total_steps = 0

for ep in range(1, EPISODES):
    state = env.reset()
    state = normalize_state(state)
    done = False
    steps = 0

    while not done:
        action = dqn_agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        score = info[0]
        next_state = normalize_state(next_state)

        if score > best_score:
            best_score = score

        steps += 1
        total_steps += 1

        if total_steps % TARGET_UPDATE == 0:
            dqn_agent.target_net.load_state_dict(dqn_agent.policy_net.state_dict())

        dqn_agent.memory.append((state, action, reward, next_state, done))
        dqn_agent.epsilon = max(EPSILON_END, dqn_agent.epsilon * EPSILON_DECAY)

        if steps % 4 == 0:
            dqn_agent.replay()

        state = next_state

        total_reward += reward
    avg_score += score

    if ep % 50 == 0:
        avg_reward = total_reward / 50
        avg_score_val = avg_score / 50

        print(
            f"Ep {ep:5d} | avg reward: {avg_reward:7.1f} | steps: {steps:4d} | "
            f"best score: {best_score:3d} | avg score: {avg_score_val:.2f} | "
            f"epsilon: {dqn_agent.epsilon:.4f} | "
            f"lr: {dqn_agent.optimizer.param_groups[0]['lr']:.6f} | "
            f"loss: {dqn_agent.get_loss(reset=True):.5f}"
        )

        best_score = 0
        total_reward = 0
        avg_score = 0
        torch.save(dqn_agent.policy_net.state_dict(), f"models/flappy_dqn_model_{ep}.pth")

torch.save(dqn_agent.policy_net.state_dict(), "models/flappy_dqn_model_final.pth")
