from dataclasses import dataclass, field
from pathlib import Path

from coordinator.ladder import flat_ladder, validate_levels

AGENT_KINDS = ("claude", "haar", "scripted")


@dataclass(frozen=True)
class AgentSpec:
    name: str
    kind: str                 # claude: LLM agent; haar: brute-force search (no LLM); scripted: tests only
    prompt: Path | None = None  # the agent's .md (system prompt), for kind == "claude"
    mode: str | None = None     # behavior, for kind == "scripted"

    def __post_init__(self):
        if self.kind not in AGENT_KINDS:
            raise ValueError(f"unknown agent kind {self.kind!r}")
        if self.kind == "claude" and (self.prompt is None or not Path(self.prompt).is_file()):
            raise ValueError(f"agent {self.name}: prompt file {self.prompt} not found")
        if not self.name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"agent name {self.name!r} must be letters, digits, - or _")


@dataclass
class Config:
    rounds: int = 10
    agents: list = field(default_factory=list)
    ladder: list = field(default_factory=flat_ladder)    # 0.30, 0.33, ..., 0.48, 0.49
    fails_per_level: int = 5
    carry_over: bool = True
    deadline_s: float = 40.0          # per attempt; agents still running are killed
    ns_floor: float = 1.0             # extra ns below this count as this (timing resolution)
    timing_images: int = 256          # small, so the images stay in cache (see harness.build.marginal_ns)
    timing_rounds: int = 3
    timing_reps: int = 7
    fallback: bool = True             # brute-force Haar search when the ladder is exhausted
    confidence: float = 0.0           # pass only if val_err + confidence * standard error < bar (0 = plain)
    weight_cap: float | None = None   # cap each weight at weight_cap / n after each update (None = AdaBoost)
    balanced_init: bool = False       # start with half the weight on faces, half on non-faces
    max_llm_sessions: int = 300       # stop the run before starting more LLM agent sessions than this
    max_skips_in_a_row: int = 2       # stop the run: a skipped round leaves the weights, so it repeats
    max_failing_attempts: int = 2     # stop the run: Claude sessions failed (usage limit, API) this many attempts in a row
    model: str = "claude-sonnet-5-5"
    effort: str = "low"
    max_turns: int = 8
    seed: int = 0

    def __post_init__(self):
        validate_levels(self.ladder)
        if self.rounds < 1 or self.fails_per_level < 1 or self.max_turns < 1:
            raise ValueError("rounds, fails_per_level and max_turns must be positive")
        if self.confidence < 0 or (self.weight_cap is not None and self.weight_cap <= 1):
            raise ValueError("confidence must be >= 0 and weight_cap > 1")
        if not self.deadline_s > 0 or self.ns_floor <= 0:
            raise ValueError("deadline and ns_floor must be positive")
        if not self.agents:
            raise ValueError("at least one agent is required")
        names = [a.name for a in self.agents]
        if len(set(names)) != len(names):
            raise ValueError("agent names must be unique")
