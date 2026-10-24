"""The maths Vertiq learns with: a logistic model over named features, its scores, and the
statistics the insights are tested with. Pure Python on purpose: the models are small (a few dozen
features, tens of thousands of rows), so no new dependency is needed and they train in seconds.

A model is a plain dict (`to_dict`), so it is stored as JSON (models.VertiqModel) and loaded anywhere.
"""
from __future__ import annotations

import math
import operator
import random

_EPS = 1e-9
Z_CLIP = 6.0      # a customer unlike anyone in training counts as at most 6 spreads away, never millions


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


class Logistic:
    """Logistic regression on {feature name: value} rows. Features are standardised with the
    training mean and spread, L2-regularised, and fitted by mini-batch gradient descent with
    AdaGrad steps. Unknown features at scoring time are ignored; missing ones count as the mean."""

    def __init__(self, names: list[str] | None = None):
        self.names: list[str] = list(names or [])
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}
        self.w: dict[str, float] = {}
        self.b: float = 0.0
        self.offsets: dict[str, float] = {}      # a learned shift per group (business), on the logit

    # ---- fitting -------------------------------------------------------------------------------
    def _standardise(self, rows: list[dict]) -> None:
        if not self.names:
            self.names = sorted({k for r in rows for k in r})
        n = max(1, len(rows))
        for k in self.names:
            vals = [float(r.get(k, 0.0)) for r in rows]
            m = sum(vals) / n
            var = sum((v - m) ** 2 for v in vals) / n
            sd = math.sqrt(var)
            self.mean[k], self.std[k] = m, (sd if sd > 1e-6 else 1.0)

    def _zk(self, row: dict, k: str) -> float:
        z = (float(row.get(k, self.mean[k])) - self.mean[k]) / self.std[k]
        return max(-Z_CLIP, min(Z_CLIP, z))

    def _cols(self) -> list[tuple[str, float, float]]:
        """(name, mean, spread) per feature, cached; rebuilt whenever fit or from_dict sets them."""
        c = self.__dict__.get("_cols_cache")
        if c is None:
            c = self._cols_cache = [(k, self.mean[k], self.std[k]) for k in self.names]
        return c

    def _z(self, row: dict) -> list[float]:
        # the same arithmetic as _zk for each name, without a method call and three dict lookups per value
        get = row.get
        return [max(-Z_CLIP, min(Z_CLIP, (float(get(k, m)) - m) / sd)) for k, m, sd in self._cols()]

    def _link(self, z: float) -> float:
        return sigmoid(z)

    def _start(self, y: list) -> float:
        base = min(1 - 1e-3, max(1e-3, sum(y) / len(y)))
        return logit(base)                       # start at the base rate: fewer steps to converge

    def fit(self, rows: list[dict], y: list, *, l2: float = 1e-3, epochs: int = 12,
            lr: float = 0.3, batch: int = 256, seed: int = 7) -> "Logistic":
        if len(rows) != len(y) or not rows:
            raise ValueError("need matching, non-empty rows and labels")
        self._standardise(rows)
        self.__dict__.pop("_cols_cache", None)
        X = [self._z(r) for r in rows]
        d = len(self.names)
        w = [0.0] * d
        b = self._start(y)
        gw, gb = [_EPS] * d, _EPS
        order = list(range(len(X)))
        rnd = random.Random(seed)
        link, mul = self._link, operator.mul
        for _ in range(epochs):
            rnd.shuffle(order)
            for s in range(0, len(order), batch):
                idx = order[s:s + batch]
                grad = [0.0] * d
                g0 = 0.0
                for i in idx:
                    xi = X[i]
                    # the same sums in the same order as a loop over j: results are identical, only faster
                    err = link(b + sum(map(mul, w, xi))) - y[i]
                    g0 += err
                    grad = [g + err * x for g, x in zip(grad, xi)]
                m = len(idx)
                for j in range(d):
                    g = grad[j] / m + l2 * w[j]
                    gw[j] += g * g
                    w[j] -= lr * g / math.sqrt(gw[j])
                g0 /= m
                gb += g0 * g0
                b -= lr * g0 / math.sqrt(gb)
        self.w = dict(zip(self.names, w))
        self.b = b
        return self

    def fit_offsets(self, rows: list[dict], y: list[int], groups: list[str], *, prior: float = 200.0) -> None:
        """A logit shift per group (e.g. per business) on top of the shared model, shrunk toward 0
        by `prior` pseudo-rows: a business with little history leans on what every sector taught
        the shared model, and one with a lot gets its own correction."""
        by: dict[str, list[int]] = {}
        for i, g in enumerate(groups):
            by.setdefault(g, []).append(i)
        for g, idx in by.items():
            zs = [self.raw(rows[i]) for i in idx]
            o = 0.0
            for _ in range(25):                  # 1-D Newton on the group's own log-likelihood
                ps = [sigmoid(z + o) for z in zs]
                grad = sum(p - y[i] for p, i in zip(ps, idx))
                hess = sum(p * (1 - p) for p in ps) + _EPS
                step = grad / hess
                o = max(-4.0, min(4.0, o - step))  # one-class groups would run off to infinity
                if abs(step) < 1e-6:
                    break
            self.offsets[g] = round(o * len(idx) / (len(idx) + prior), 6)

    # ---- scoring -------------------------------------------------------------------------------
    def raw(self, row: dict) -> float:
        w = self.w
        return self.b + sum(w[k] * z for (k, _m, _s), z in zip(self._cols(), self._z(row)))

    def predict(self, row: dict, group: str | None = None) -> float:
        return self._link(self.raw(row) + (self.offsets.get(group, 0.0) if group else 0.0))

    def contributions(self, row: dict) -> list[tuple[str, float]]:
        """Each feature's push on the logit for this row, largest first (signed)."""
        c = [(k, self.w[k] * self._zk(row, k)) for k in self.names]
        return sorted(c, key=lambda kv: -abs(kv[1]))

    # ---- storage -------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"names": self.names, "mean": self.mean, "std": self.std, "w": self.w, "b": self.b,
                "offsets": self.offsets}

    @classmethod
    def from_dict(cls, d: dict) -> "Logistic":
        m = cls(d.get("names") or [])
        m.mean, m.std, m.w = dict(d["mean"]), dict(d["std"]), dict(d["w"])
        m.__dict__.pop("_cols_cache", None)
        m.b, m.offsets = float(d["b"]), dict(d.get("offsets") or {})
        return m


class Linear(Logistic):
    """Ridge regression on the same named, standardised features (e.g. the log of a bill). Same
    fitting, storage and contributions; the output is the raw value, not a probability."""

    def _link(self, z: float) -> float:
        return z

    def _start(self, y: list) -> float:
        return sum(y) / len(y)

    def fit_offsets(self, rows, y, groups, *, prior: float = 200.0) -> None:
        by: dict[str, list[int]] = {}
        for i, g in enumerate(groups):
            by.setdefault(g, []).append(i)
        for g, idx in by.items():                # the mean residual, shrunk toward 0 by `prior`
            resid = sum(y[i] - self.raw(rows[i]) for i in idx)
            self.offsets[g] = round(resid / (len(idx) + prior), 6)


def r2(pred: list[float], y: list[float]) -> float | None:
    if len(y) < 2:
        return None
    m = sum(y) / len(y)
    tot = sum((v - m) ** 2 for v in y)
    return 1 - sum((p - v) ** 2 for p, v in zip(pred, y)) / tot if tot else None


# ---- scores ------------------------------------------------------------------------------------
def auc(scores: list[float], y: list[int]) -> float | None:
    """Area under the ROC curve by ranks (ties share their average rank). None with one class."""
    pos = sum(y)
    neg = len(y) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__)
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    rank_pos = sum(r for r, t in zip(ranks, y) if t)
    return (rank_pos - pos * (pos + 1) / 2) / (pos * neg)


def brier(probs: list[float], y: list[int]) -> float | None:
    return sum((p - t) ** 2 for p, t in zip(probs, y)) / len(y) if y else None


def calibration(probs: list[float], y: list[int], bins: int = 5) -> list[dict]:
    """Predicted vs actual rate in equal-count bins, the check that "70%" means 70%."""
    pairs = sorted(zip(probs, y))
    out = []
    n = len(pairs)
    for b in range(bins):
        part = pairs[b * n // bins:(b + 1) * n // bins]
        if part:
            out.append({"predicted": round(sum(p for p, _ in part) / len(part), 3),
                        "actual": round(sum(t for _, t in part) / len(part), 3), "n": len(part)})
    return out


# ---- tests for the insights -------------------------------------------------------------------
def two_proportion_p(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided p-value that two rates differ (pooled z-test). 1.0 when either group is empty."""
    if not n1 or not n2:
        return 1.0
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (x1 / n1 - x2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def benjamini_yekutieli(pvalues: list[float], q: float = 0.1) -> list[bool]:
    """False-discovery-rate control that holds under any dependence between the tests: Benjamini-
    Hochberg at q / c(m), with c(m) = 1 + 1/2 + ... + 1/m. Use it when the tests share data, such as
    overlapping segments."""
    m = len(pvalues)
    c = sum(1.0 / i for i in range(1, m + 1)) if m else 1.0
    return benjamini_hochberg(pvalues, q / c)


def benjamini_hochberg(pvalues: list[float], q: float = 0.1) -> list[bool]:
    """Which tests survive a false-discovery rate of `q` across the whole family: the more
    patterns are searched for, the stronger each one has to be before it is shown."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    cutoff = -1
    for rank, i in enumerate(order, start=1):
        if pvalues[i] <= q * rank / m:
            cutoff = rank
    keep = [False] * m
    for rank, i in enumerate(order, start=1):
        keep[i] = rank <= cutoff
    return keep
