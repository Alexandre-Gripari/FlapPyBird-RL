import numpy as np
import pygame

from .entities import PlayerMode, Score, Pipes, Player, Floor, Background
from .flappy import Flappy
from .utils.reward_config import RewardConfig


class FlappyEnv:
    """
    A Flappy Bird environment for reinforcement learning.
    The state is represented as (dx, dy, vel_y):
        dx: distance to next pipe (0 to 288)
        dy: distance to center of next pipe gap (-148 to 256)
        vel_y: player vertical velocity (-8 to 10)
    The actions are:
        0: do nothing
        1: flap
    The rewards are:
        +0.1 for each frame alive
        +0.2 for being within 30 pixels of the pipe gap center
        +5 for passing near a pipe
        +20 for passing a pipe
        -0.05 for flapping
        -10 for dying
    The game ends when the player collides with a pipe or the ground.
    """

    def __init__(self, render=False, speed=30, deep_qlearning=False):
        self.total_reward = None
        self.done = None
        self.score = None
        self.pipes = None
        self.player = None
        self.floor = None
        self.background = None
        self.last_pipe_id = None
        self.render_mode = render
        self.speed = speed
        self.game = Flappy()
        self.config = self.game.config
        self.screen = self.config.screen
        self.clock = pygame.time.Clock()
        self.reset()
        self.rewards = RewardConfig.from_mode(deep_qlearning)

    def reset(self):
        """
        Reset the environment to the initial state.
        :return: state as (dx, dy, vel_y)
        """
        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.player = Player(self.config)
        self.pipes = Pipes(self.config)
        self.score = Score(self.config)
        self.last_pipe_id = None

        self.player.set_mode(PlayerMode.NORMAL)
        self.done = False
        self.total_reward = 0

        for _ in range(35):
            self.background.tick()
            self.floor.tick()
            self.pipes.tick()

        return self._get_state()

    def _get_next_pipe(self):
        """
        :return: the next pipe (up, low)
        where:
            up: upper pipe
            low: lower pipe
        """
        for up, low in zip(self.pipes.upper, self.pipes.lower):
            if up.x + up.w > self.player.x:
                return up, low
        return self.pipes.upper[0], self.pipes.lower[0]

    def _get_prev_pipe(self):
        """
        :return: the previous pipe or None
        """
        prev_up = None
        for up in self.pipes.upper:
            if up.x + up.w < self.player.x:
                return up
        return prev_up

    def _get_prev_pipe2(self):
        """
        :return: the current pipe or None
        """
        prev_up = None
        for up in self.pipes.upper:
            if up.x + 17 < self.player.x:
                return up
        return prev_up

    def _get_state(self):
        """
        :return: actual state as (dx, dy, vel_y)
        where:
            dx: distance to next pipe (0 to 288)
            dy: distance to center of next pipe gap (-148 to 256)
            vel_y: player vertical velocity (-8 to 10)
        """
        up, low = self._get_next_pipe()

        screen_width = self.config.window.width
        screen_height = self.config.window.height

        dx = up.x - self.player.x
        dx = max(0, min(screen_width, dx))

        gap_center_y = up.y + up.h + self.pipes.pipe_gap / 2
        dy = gap_center_y - self.player.y
        dy = max(-screen_height // 2, min(screen_height // 2, dy))

        return np.array([int(dx), int(dy), int(self.player.vel_y)], dtype=np.int32)

    def compute_reward(self, action: int, prev_pipe, prev_pipe2, done: bool, dy: float) -> float:
        r = self.rewards.frame_alive
        if action == 1:
            r += self.rewards.flap
        if -30 < dy < 30:
            r += self.rewards.near_pipe_gap
        if prev_pipe and id(prev_pipe) != self.last_pipe_id:
            r += self.rewards.pass_pipe
        elif prev_pipe2 and id(prev_pipe2) != self.last_pipe_id:
            r += self.rewards.pass_near_pipe
        if done:
            r += self.rewards.die
        return r

    def step(self, action):
        """
        Make a step in the environment.
        :param: action: 0 (no flap) or 1 (flap)
        :return: state, reward, done
        where:
            state: (dx, dy, vel_y)
            reward: total reward for this step
            done: True if the game is over
        """


        if action == 1:
            self.player.flap()

        self.background.tick()
        self.floor.tick()
        self.pipes.tick()
        self.score.tick()
        self.player.tick()

        self.done = self.player.collided(self.pipes, self.floor)

        prev_pipe = self._get_prev_pipe()
        prev_pipe2 = self._get_prev_pipe2()

        up, low = self._get_next_pipe()

        screen_height = self.config.window.height

        gap_center_y = up.y + up.h + self.pipes.pipe_gap / 2
        dy = gap_center_y - self.player.y
        dy = max(-screen_height // 2, min(screen_height // 2, dy))

        state = self._get_state()

        reward = self.compute_reward(action, prev_pipe, prev_pipe2, self.done, dy)

        if self.render_mode:
            self._render_frame()

        return state, reward, self.done, self.score.score

    def _render_frame(self):
        """Renders the current game state to the screen"""
        self.background.draw()
        for up, low in zip(self.pipes.upper, self.pipes.lower):
            up.draw()
            low.draw()
        self.floor.draw()
        self.player.draw()
        self.score.draw()
        pygame.display.update()
        self.clock.tick(self.speed)

    def close(self):
        """Closes the game window"""
        pygame.quit()
