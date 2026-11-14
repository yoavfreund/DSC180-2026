"""The error bar: how good a weak rule must be before it is accepted.

The bar starts strict. After `fails_per_level` failed attempts in a row it
moves up one level. When a round starts, the bar moves DOWN one level from
where the last round ended (carry-over), so each round first tries a slightly
stricter bar than the last one that worked. Weighted error drifts toward 0.5 as
boosting goes on, so restarting at the bottom every round would waste attempts.
"""


def flat_ladder(start=0.30, step=0.03, cap=0.49):
    """start, start+step, ... and finally cap. Example: 0.30, 0.33, ..., 0.48, 0.49."""
    levels, v = [], start
    while v < cap - 1e-9:
        levels.append(round(v, 6))
        v += step
    levels.append(cap)
    return levels


def parse_ladder(text):
    levels = [float(v) for v in text.split(",") if v.strip()]
    validate_levels(levels)
    return levels


def validate_levels(levels):
    if not levels:
        raise ValueError("the ladder needs at least one level")
    if any(not 0 < v < 0.5 for v in levels):
        raise ValueError("every ladder level must be between 0 and 0.5")
    if any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError("ladder levels must increase")


class Ladder:
    def __init__(self, levels, fails_per_level, carry_over=True):
        validate_levels(levels)
        if fails_per_level < 1:
            raise ValueError("fails_per_level must be at least 1")
        self.levels = list(levels)
        self.fails_per_level = fails_per_level
        self.carry_over = carry_over
        self.position = 0
        self.fails = 0
        self._started = False

    @property
    def bar(self):
        return self.levels[self.position]

    @property
    def at_top(self):
        return self.position == len(self.levels) - 1

    def start_round(self):
        if self._started and self.carry_over:
            self.position = max(self.position - 1, 0)
        elif not self.carry_over:
            self.position = 0
        self._started = True
        self.fails = 0

    def record_failure(self):
        """Count one failed attempt. Returns 'failed', 'raised' or 'exhausted'."""
        self.fails += 1
        if self.fails < self.fails_per_level:
            return "failed"
        if self.at_top:
            return "exhausted"
        self.position += 1
        self.fails = 0
        return "raised"
