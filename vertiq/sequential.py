"""Sequential testing: monitor a binary outcome as it arrives (a conversion, a failure, an on-time
delivery) and stop as soon as the evidence is strong enough, without the inflated false-alarm rate of
checking a fixed-sample test repeatedly.

Wald's sequential probability ratio test compares H0 (rate = theta0, the baseline) with H1 (rate =
theta1, the smallest rate worth adopting). The boundaries keep the type I and type II error rates
approximately at `alpha` and `beta`. Pass the result to the evaluation gate as `evidence`: evidence for
H0 blocks a promotion; evidence never forces one.
"""
from __future__ import annotations

import math


class SequentialTest:
    """Wald's SPRT for a Bernoulli rate. `theta1` may be above or below `theta0` (an improvement can be
    a higher conversion rate or a lower failure rate)."""

    def __init__(self, theta0: float, theta1: float, alpha: float = 0.05, beta: float = 0.2):
        if not (0 < theta0 < 1 and 0 < theta1 < 1) or theta0 == theta1:
            raise ValueError("require 0 < theta0, theta1 < 1 and theta0 != theta1")
        self.theta0, self.theta1, self.alpha, self.beta = theta0, theta1, alpha, beta
        self.win, self.loss = math.log(theta1 / theta0), math.log((1 - theta1) / (1 - theta0))
        self.upper, self.lower = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
        self.llr, self.n, self.successes, self.decision = 0.0, 0, 0, "continue"

    def update(self, outcome: int) -> str:
        """Add one observation (1 = the event happened). Returns "continue", "accept_h1" or "accept_h0"."""
        if self.decision != "continue":
            return self.decision
        self.n += 1
        self.successes += int(bool(outcome))
        self.llr += self.win if outcome else self.loss
        if self.llr >= self.upper:
            self.decision = "accept_h1"
        elif self.llr <= self.lower:
            self.decision = "accept_h0"
        return self.decision

    def run(self, outcomes) -> str:
        for o in outcomes:
            if self.update(o) != "continue":
                break
        return self.decision

    def state(self) -> dict:
        return {"decision": self.decision, "n": self.n, "observed_rate": round(self.successes / self.n, 4) if self.n else None,
                "log_likelihood_ratio": round(self.llr, 4), "bounds": [round(self.lower, 4), round(self.upper, 4)],
                "theta0": self.theta0, "theta1": self.theta1, "alpha": self.alpha, "beta": self.beta}
