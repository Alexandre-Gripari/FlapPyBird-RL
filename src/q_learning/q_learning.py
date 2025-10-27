from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np

from .utils import get_q, soft_discretize, get_learning_rate
from ..flappy_env import FlappyEnv


@dataclass
class TrainingConfig:
    """Configuration for Q-learning training"""
    # Training hyperparameters
    episodes: int = 50000
    alpha: float = 1.0
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.005
    epsilon_decay_power: float = 1.0
    use_linear_decay: bool = True

    # Environment parameters
    render: bool = True
    speed: int = 0

    # State space boundaries
    dx_min: float = 0
    dx_max: float = 212
    dy_min: float = -104
    dy_max: float = 256
    vel_y_min: float = -8
    vel_y_max: float = 10

    # Discretization bins
    dx_bins: int = 24
    dy_bins: int = 36
    vel_y_bins: int = 5

    # Actions
    actions: tuple = (0, 1)

    # Logging and saving
    log_interval: int = 50
    save_path: Optional[str] = "q_matrix.npy"


class QLearningTrainer:
    """Q-Learning trainer for Flappy Bird environment"""

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the Q-Learning trainer
        :param config: Training configuration
        :return: None
        """
        self.config = config or TrainingConfig()
        self._init_config()
        self.env = FlappyEnv(render=self.config.render, speed=self.config.speed)

        self.q: Optional[np.ndarray] = None
        self._initialize_q_table()

        self.best_score = 0
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_scores = []

    def _initialize_q_table(self):
        """Initialize Q-table and state-action counts"""
        self.q = np.zeros((self.config.dx_bins, self.config.dy_bins, self.config.vel_y_bins, len(self.config.actions)),
                          dtype=np.float32)

    def _init_config(self):
        """
        Initialize computed properties for the config
        """
        self.total_states = self.config.dx_bins * self.config.dy_bins * self.config.vel_y_bins * len(
            self.config.actions)
        self.state_action_counts = np.zeros(
            (self.config.dx_bins, self.config.dy_bins, self.config.vel_y_bins, len(self.config.actions)),
            dtype=np.int32)

    def update_config(self, **kwargs):
        """
        Update training configuration parameters dynamically
        :param kwargs: Configuration parameters to update
        :return: None
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Invalid config parameter: {key}")

    def get_epsilon(self, episode: int) -> float:
        """
        Calculate epsilon value for current episode
        :param episode: Current episode number
        :return: Epsilon value
        """
        if self.config.use_linear_decay:
            return max(self.config.epsilon_end,
                       self.config.epsilon_start - (self.config.epsilon_start - self.config.epsilon_end) * (
                               episode / self.config.episodes))
        else:
            return (self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * (
                    (1 - episode / self.config.episodes) ** self.config.epsilon_decay_power))

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Select action using epsilon-greedy policy
        :param state: current state (dx, dy, vel_y)
        :param epsilon: current epsilon value
        :return: action (0 or 1)
        """
        if np.random.rand() < epsilon:
            return np.random.randint(0, len(self.config.actions))
        else:
            q_values = get_q(state, self.q, self.config)
            return np.argmax(q_values)

    def update_q(self, state, action, reward, next_state):
        """
        Update Q-value using the Q-learning formula.
        Q(s,a) = Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))
        :param state: current state (dx, dy, vel_y)
        :param action: action taken (0 or 1)
        :param reward: reward received
        :param next_state: next state (dx', dy', vel_y')
        :return: None
        """
        dx_bin, dy_bin, vy_bin = soft_discretize(state, self.config)
        next_dx_bin, next_dy_bin, next_vy_bin = soft_discretize(next_state, self.config)

        alpha = get_learning_rate(dx_bin, dy_bin, vy_bin, action, self.state_action_counts)
        self.state_action_counts[dx_bin, dy_bin, vy_bin, action] += 1

        current_q = self.q[dx_bin, dy_bin, vy_bin, action]
        max_next_q = np.max(self.q[next_dx_bin, next_dy_bin, next_vy_bin, :])

        self.q[dx_bin, dy_bin, vy_bin, action] = current_q + alpha * (
                    reward + self.config.gamma * max_next_q - current_q)

    def train_episode(self, episode: int) -> Tuple[float, int, int]:
        """
        Train for a single episode
        :param episode: Current episode number
        :return: total_reward, steps, score
        """
        state = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        info = {}

        epsilon = self.get_epsilon(episode)

        while not done:
            action = self.select_action(state, epsilon)
            next_state, reward, done, info = self.env.step(action)

            steps += 1
            total_reward += reward

            self.update_q(state, action, reward, next_state)

            state = next_state

        score = info[0] if isinstance(info, tuple) else info.get('score', 0)
        self.best_score = max(self.best_score, score)

        return total_reward, steps, score

    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get current training statistics
        :return: Dictionary of training statistics
        """
        visited_states = np.count_nonzero(np.any(self.state_action_counts > 0, axis=3))
        explored_percent = visited_states / self.total_states * 100
        state_action_pairs = np.sum(self.state_action_counts > 0)

        return {'visited_states': visited_states, 'explored_percent': explored_percent,
                'state_action_pairs': state_action_pairs, 'best_score': self.best_score,
                'total_states': self.total_states}

    def train(self, resume: bool = False, initial_q: Optional[np.ndarray] = None):
        """
        Run the full training loop
        :param resume: whether to resume from an existing Q-table
        :param initial_q: initial Q-table to load if resuming
        :return: None
        """
        if resume and initial_q is not None:
            self.q = initial_q.copy()

        print(f"Total states: {self.total_states:,}")
        print(f"Training for {self.config.episodes} episodes")
        print(f"Alpha: {self.config.alpha}, Epsilon: {self.config.epsilon_start} -> {self.config.epsilon_end}")

        for ep in range(self.config.episodes):
            total_reward, steps, score = self.train_episode(ep)

            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.episode_scores.append(score)

            if ep % self.config.log_interval == 0:
                stats = self.get_training_stats()
                epsilon = self.get_epsilon(ep)

                print(f"Ep {ep:5d} | reward: {total_reward:7.1f} | steps: {steps:4d} | "
                      f"score: {score:3d} | Visited: {stats['visited_states']:5d} "
                      f"({stats['explored_percent']:5.2f}%) | ε: {epsilon:.4f}")

        if self.config.save_path:
            np.save(self.config.save_path, self.q)
            print(f"\nQ-table saved to {self.config.save_path}")

        stats = self.get_training_stats()
        print(f"\nTraining complete!")
        print(f"Best score: {self.best_score}")
        print(f"States visited: {stats['visited_states']:,} / {stats['total_states']:,}")
        print(f"State-action pairs explored: {stats['state_action_pairs']:,}")

    def load_q_table(self, path: str):
        """
        Load Q-table from file
        :param path: Path to Q-table file
        :return: None
        """
        self.q = np.load(path)
        print(f"Q-table loaded from {path}")

    def save_q_table(self, path: Optional[str] = None):
        """
        Save Q-table to file
        :param path: Path to save Q-table file
        :return: None
        """
        save_path = path or self.config.save_path
        if save_path:
            np.save(save_path, self.q)
            print(f"Q-table saved to {save_path}")


if __name__ == "__main__":
    trainer = QLearningTrainer()
    trainer.train()
