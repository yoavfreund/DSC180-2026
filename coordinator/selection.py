"""Choose among rules under the bar: the most boosting progress per nanosecond.

AdaBoost multiplies a bound on training error by sqrt(1 - 4 g^2) per round,
where g = 0.5 - error is the rule's edge. So one rule's progress is

    progress = -0.5 * ln(1 - 4 g^2)

and the coordinator picks the rule with the largest progress / ns. Speed only
decides WHICH rule is picked. The rule's alpha comes from its error alone.
"""
import math

ERR_FLOOR = 1e-6


def progress(err):
    err = min(max(err, ERR_FLOOR), 0.5)
    g = 0.5 - err
    return -0.5 * math.log(1.0 - 4.0 * g * g)


def progress_per_ns(err, ns, ns_floor=1.0):
    return progress(err) / max(ns, ns_floor)


def alpha(err):
    e = min(max(err, ERR_FLOOR), 1.0 - ERR_FLOOR)
    return 0.5 * math.log((1.0 - e) / e)


def required_error(target_ppn, ns, ns_floor=1.0):
    """Highest error a rule taking `ns` could have and still reach target_ppn. None if no error is low enough."""
    p = target_ppn * max(ns, ns_floor)
    if p >= progress(ERR_FLOOR):
        return None
    g2 = (1.0 - math.exp(-2.0 * p)) / 4.0
    return 0.5 - math.sqrt(g2)


def choose(candidates, ns_floor=1.0):
    """Highest progress per ns; ties go to the lower error, then the earlier candidate."""
    return max(candidates, key=lambda c: (progress_per_ns(c.val_err, c.ns, ns_floor), -c.val_err, -c.order))
