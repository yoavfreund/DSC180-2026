from dataclasses import dataclass


@dataclass
class Config:
    T: int = 10
    K: int = 3
    max_refine: int = 3
    gamma: float = 0.01
    val_ratio: float = 0.67
    combiner: str = "adaboost_weighted"
    seed: int = 42
    time_limit: float = 55.0
    speed_tolerance: float = 0.005
    llm_adapter: str | None = None

    def __post_init__(self):
        if self.T < 1 or self.K < 1 or self.max_refine < 1:
            raise ValueError("rounds, candidates, and attempts must be positive")
        if not 0 < self.val_ratio < 1 or not 0 <= self.gamma < .5:
            raise ValueError("invalid validation ratio or margin")
        if not 0 < self.time_limit < 60 or self.speed_tolerance < 0:
            raise ValueError("time limit must be under 60 seconds and tolerance nonnegative")
