from dataclasses import dataclass


@dataclass
class RewardConfig:
    frame_alive: float = 0.1
    near_pipe_gap: float = 0.2
    pass_near_pipe: float = 5.0
    pass_pipe: float = 20.0
    flap: float = -0.05
    die: float = -10.0

    @classmethod
    def from_mode(cls, deep_qlearning: bool) -> "RewardConfig":
        if deep_qlearning:
            return cls(frame_alive=0.01,
                       near_pipe_gap=0.05,
                       pass_near_pipe=0.5,
                       pass_pipe=2.0,
                       flap=-0.005,
                       die=-10.0)
        return cls()
