"""Sequential testing: monitor a binary outcome as it arrives (a conversion, a failure, an on-time
delivery) and stop as soon as the evidence is strong enough, without the inflated false-alarm rate of
checking a fixed-sample test repeatedly.

Wald's sequential probability ratio test compares H0 (rate = theta0, the baseline) with H1 (rate =
theta1, the smallest rate worth adopting). The boundaries keep the type I and type II error rates
approximately at `alpha` and `beta`. Pass the result to the evaluation gate as `evidence`: evidence for
H0 blocks a promotion; evidence never forces one.

`TwoSampleTest` compares two live arms (the current model or process A against a candidate B) when neither
rate is known in advance: a mixture sequential probability ratio test (mSPRT) with an always-valid p-value,
so the result can be checked after every observation and the false-alarm rate stays at `alpha`.
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


class TwoSampleTest:
    """Mixture SPRT for the difference between two arms (B minus A) of a binary or numeric outcome.

    The observed difference d, with variance V = s2 (1/n_a + 1/n_b) from the pooled variance s2, is
    compared with H0 (difference = `theta0`) against a normal mixture of alternatives with standard
    deviation `tau` (set it near the smallest difference worth detecting). The likelihood ratio is
        L = sqrt(V / (V + tau^2)) * exp(tau^2 (d - theta0)^2 / (2 V (V + tau^2)))
    and the always-valid p-value is the running minimum of 1 / L (Johari, Pekelis and Walsh, 2017).
    B is declared different from A when it falls below `alpha`. The test does not stop for "no
    difference"; `max_n` per arm turns that into a decision ("accept_h0") when the budget runs out."""

    def __init__(self, tau: float, alpha: float = 0.05, theta0: float = 0.0, max_n: int | None = None, min_n: int = 20):
        if tau <= 0 or not 0 < alpha < 1:
            raise ValueError("require tau > 0 and 0 < alpha < 1")
        self.tau, self.alpha, self.theta0, self.max_n, self.min_n = tau, alpha, theta0, max_n, min_n
        self.n = {"a": 0, "b": 0}
        self.sum = {"a": 0.0, "b": 0.0}
        self.sq = {"a": 0.0, "b": 0.0}
        self.p_value, self.decision = 1.0, "continue"

    def update(self, arm: str, outcome: float) -> str:
        """Add one observation to arm "a" or "b". Returns "continue", "accept_h1" or "accept_h0"."""
        if arm not in self.n:
            raise ValueError('arm must be "a" or "b"')
        if self.decision != "continue":
            return self.decision
        x = float(outcome)
        self.n[arm] += 1
        self.sum[arm] += x
        self.sq[arm] += x * x
        na, nb = self.n["a"], self.n["b"]
        if min(na, nb) >= self.min_n:
            d = self.difference()
            mean = (self.sum["a"] + self.sum["b"]) / (na + nb)
            s2 = max((self.sq["a"] + self.sq["b"]) / (na + nb) - mean * mean, 1e-12)
            v = s2 * (1 / na + 1 / nb)
            t2 = self.tau ** 2
            log_l = 0.5 * math.log(v / (v + t2)) + t2 * (d - self.theta0) ** 2 / (2 * v * (v + t2))
            self.p_value = min(self.p_value, math.exp(-log_l) if log_l < 700 else 0.0)
            if self.p_value <= self.alpha:
                self.decision = "accept_h1"
        if self.decision == "continue" and self.max_n and min(na, nb) >= self.max_n:
            self.decision = "accept_h0"
        return self.decision

    def difference(self) -> float:
        na, nb = self.n["a"], self.n["b"]
        return (self.sum["b"] / nb if nb else 0.0) - (self.sum["a"] / na if na else 0.0)

    def run(self, pairs) -> str:
        """Feed (arm, outcome) pairs until a decision."""
        for arm, o in pairs:
            if self.update(arm, o) != "continue":
                break
        return self.decision

    def state(self) -> dict:
        return {"decision": self.decision, "n": self.n["a"] + self.n["b"], "n_a": self.n["a"], "n_b": self.n["b"],
                "difference": round(self.difference(), 4), "p_value": round(self.p_value, 5),
                "tau": self.tau, "alpha": self.alpha, "theta0": self.theta0}
