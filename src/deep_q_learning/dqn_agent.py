import random
from collections import deque

import numpy as np
import torch

from .q_network import QNetwork


class DQNAgent:
    """
    Deep Q-Learning Agent
    ----------------------------------
    Implements a Deep Q-Learning agent with experience replay and target network.
    Uses an epsilon-greedy policy for action selection.
    1. Initializes policy and target networks.
    2. Stores experiences in a replay buffer.
    3. Samples mini-batches from the replay buffer to train the policy network.
    4. Updates the target network periodically.
    5. Uses Smooth L1 Loss (Huber Loss) for training stability.
    6. Clips gradients to prevent exploding gradients.
    7. Tracks average loss over training iterations.
    8. Supports GPU acceleration if available.
    :param input_size: Size of the input state
    :param output_size: Number of possible actions
    :param epsilon: Initial exploration rate for epsilon-greedy policy
    :param gamma: Discount factor for future rewards
    :param lr: Learning rate for the optimizer
    :param batch_size: Mini-batch size for training
    """

    def __init__(self, input_size, output_size, epsilon: float = 1,
                 gamma: float = 0.99,
                 lr: float = 1e-3,
                 batch_size=32
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.output_size = output_size

        self.epsilon = epsilon
        self.gamma = gamma
        self.memory_size = 100000
        self.batch_size = batch_size

        self.policy_net = QNetwork(input_size, output_size).to(self.device)
        self.target_net = QNetwork(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = deque(maxlen=self.memory_size)

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=lr
        )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9995)

        self.loss_fct = torch.nn.SmoothL1Loss()

        self.loss_total = 0
        self.loss_count = 0

    def get_action(self, state):
        """
        Select an action using epsilon-greedy policy
        :param state: Current state
        :return: Selected action
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.output_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    def step_scheduler(self):
        for lr in self.optimizer.param_groups:
            if lr['lr'] < 1e-6:
                return
        if self.scheduler:
            self.scheduler.step()

    def replay(self):
        """
        Sample a mini-batch from memory and perform a training step
        :return: None
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)

        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.loss_fct(current_q_values, target_q_values)

        self.loss_total += loss.item()
        self.loss_count += 1

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

    def get_loss(self, reset=False):
        """
        Get the average loss over training iterations
        :param reset: Whether to reset the loss tracking after getting the value
        :return: Average loss
        """
        if self.loss_count == 0:
            return 0.0
        loss = self.loss_total / self.loss_count
        if reset:
            self.loss_count = 0
            self.loss_total = 0
        return loss
