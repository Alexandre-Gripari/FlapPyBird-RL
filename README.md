[FlapPyBird](https://sourabhv.github.io/FlapPyBird)
===============

A Flappy Bird Clone made using [python-pygame][pygame]

> If you are in interested in the old one-file code for the game, you can [find it here][one-file-game]

[pygame]: http://www.pygame.org
[one-file-game]: https://github.com/sourabhv/FlapPyBird/blob/038359dc6122f8d851e816ddb3e7d28229d585e5/flappy.py


Setup (as tested on MacOS)
---------------------------

1. Install Python 3 from [here](https://www.python.org/download/releases/) (or use brew/apt/pyenv)

2. Run `make init` (this will install pip packages, use virtualenv or something similar if you don't want to install globally)

3. Run `make` to run the game. Run `DEBUG=True make` to see rects and coords

4. Use <kbd>&uarr;</kbd> or <kbd>Space</kbd> key to play and <kbd>Esc</kbd> to close the game.

5. Optionally run `make web` to run the game in the browser (`pygbag`).


# Flappy Bird Agent Instructions

This repository is a fork of a Flappy Bird clone that includes implementations of Q-Learning and Deep Q-Learning agents. Below are instructions on how to train and run these agents.

When using Q-Learning, ensure that the discretization parameters in the agent match those used during training.

## Q-Learning

### Start Training
```bash
python -m src.q_learning.q_learning
````

### Run the Agent

```bash
python -m src.q_learning.agent_q --q_matrix_path=PATH_TO_Q_MATRIX.npy
```

### Benchmarking

To benchmark the trained agent, run:

```bash
python -m src.q_learning.benchmark --q_matrix_path=PATH_TO_Q_MATRIX.npy --benchmark
```

### Modify Parameters

All parameters (hyperparameters, state discretization, number of episodes) are defined in a dataclass called `TrainingConfig`. You can either modify the default values directly or create an instance of this class with your desired parameters.

---

## Deep-Q Learning

### Start Training

```bash
python -m src.deep_q_learning.deep_q_learning
```

### Run the Agent

```bash
python -m src.deep_q_learning.agent_deep_q_learning
```

Demo
----------

https://github.com/Alexandre-Gripari/FlapPyBird-RL/blob/master/videos/deep_q_learning_100.mp4
