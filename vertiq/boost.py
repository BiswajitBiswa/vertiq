"""Gradient-boosted decision trees in pure Python: a learner for signals a linear model cannot express,
such as thresholds and interactions ("risk rises only when maintenance was skipped *and* the asset is
old"). Registered as "boosted" (classification, logistic loss) and "boosted_regression" (squared loss).

Each tree is fitted to the gradient of the loss with second-order (Newton) leaf values, on quantile
bins of each feature, with row subsampling. Explanations follow each tree's path: a feature's
contribution is the change in the node value at every split on that feature (the Saabas method), so
a score lists what raised and lowered it, as the linear models do.

A model is a plain dict (`to_dict`), like every other Vertiq model.
"""
from __future__ import annotations

import bisect
import math
import random

from .learn import Linear, Logistic, logit, sigmoid

MAX_BINS = 16


class BoostedClassifier:
    """Boosted trees on {feature name: value} rows, logistic loss. Output: a probability."""

    loss = "logistic"

    def __init__(self, names: list[str] | None = None, *, trees: int = 80, depth: int = 3, rate: float = 0.1,
                 min_leaf: int = 20, l2: float = 1.0, subsample: float = 0.8, seed: int = 7):
        self.names: list[str] = list(names or [])
        self.trees_n, self.depth, self.rate, self.min_leaf, self.l2 = trees, depth, rate, min_leaf, l2
        self.subsample, self.seed = subsample, seed
        self.cuts: dict[str, list[float]] = {}
        self.fill: dict[str, float] = {}
        self.base: float = 0.0
        self.trees: list[dict] = []
        self.offsets: dict[str, float] = {}

    # ---- fitting -------------------------------------------------------------------------------
    def _start(self, y: list) -> float:
        return logit(min(1 - 1e-3, max(1e-3, sum(y) / len(y))))

    def _grad(self, raw: float, y: float) -> tuple[float, float]:
        p = sigmoid(raw)
        return p - y, max(p * (1 - p), 1e-6)

    def _value(self, raw: float) -> float:
        return sigmoid(raw)

    def _bins(self, rows: list[dict]) -> list[list[int]]:
        if not self.names:
            self.names = sorted({k for r in rows for k in r})
        cols = []
        for k in self.names:
            vals = sorted(float(r.get(k, 0.0)) for r in rows)
            self.fill[k] = vals[len(vals) // 2]
            uniq = sorted(set(vals))
            if len(uniq) <= MAX_BINS:
                cuts = uniq[:-1]
            else:
                cuts = sorted({vals[(i * len(vals)) // MAX_BINS] for i in range(1, MAX_BINS)})
            self.cuts[k] = cuts
            cols.append([bisect.bisect_left(cuts, float(r.get(k, self.fill[k]))) for r in rows])
        return cols

    def _tree(self, cols, idx, g, h, depth) -> dict:
        G, H = sum(g[i] for i in idx), sum(h[i] for i in idx)
        node = {"v": -G / (H + self.l2)}
        if depth == self.depth or len(idx) < 2 * self.min_leaf:
            return node
        parent = G * G / (H + self.l2)
        best = (1e-9, None, None)
        for j, col in enumerate(cols):
            nb = len(self.cuts[self.names[j]]) + 1
            if nb < 2:
                continue
            gb, hb, cb = [0.0] * nb, [0.0] * nb, [0] * nb
            for i in idx:
                b = col[i]
                gb[b] += g[i]
                hb[b] += h[i]
                cb[b] += 1
            gl = hl = 0.0
            cl = 0
            for b in range(nb - 1):
                gl, hl, cl = gl + gb[b], hl + hb[b], cl + cb[b]
                cr = len(idx) - cl
                if cl < self.min_leaf or cr < self.min_leaf:
                    continue
                gain = gl * gl / (hl + self.l2) + (G - gl) ** 2 / (H - hl + self.l2) - parent
                if gain > best[0]:
                    best = (gain, j, b)
        _gain, j, b = best
        if j is None:
            return node
        left = [i for i in idx if cols[j][i] <= b]
        right = [i for i in idx if cols[j][i] > b]
        node.update(f=self.names[j], t=self.cuts[self.names[j]][b], j=j, b=b,
                    l=self._tree(cols, left, g, h, depth + 1), r=self._tree(cols, right, g, h, depth + 1))
        return node

    @staticmethod
    def _walk_bins(node: dict, cols, i: int) -> float:
        while "f" in node:
            node = node["l"] if cols[node["j"]][i] <= node["b"] else node["r"]
        return node["v"]

    def fit(self, rows: list[dict], y: list, **_unused) -> "BoostedClassifier":
        if len(rows) != len(y) or not rows:
            raise ValueError("need matching, non-empty rows and labels")
        cols = self._bins(rows)
        n = len(rows)
        self.base = self._start(y)
        raw = [self.base] * n
        rnd = random.Random(self.seed)
        self.trees = []
        for _ in range(self.trees_n):
            gh = [self._grad(raw[i], y[i]) for i in range(n)]
            g, h = [a for a, _ in gh], [b for _, b in gh]
            idx = [i for i in range(n) if rnd.random() < self.subsample] or list(range(n))
            tree = self._tree(cols, idx, g, h, 0)
            self.trees.append(tree)
            for i in range(n):
                raw[i] += self.rate * self._walk_bins(tree, cols, i)
        for t in self.trees:                     # binned indices are a training aid only
            _strip(t)
        return self

    def fit_offsets(self, rows: list[dict], y: list, groups: list[str], *, prior: float = 200.0) -> None:
        """A shift per group on the raw scale, shrunk toward 0 (as the linear models do)."""
        Logistic.fit_offsets(self, rows, y, groups, prior=prior)

    # ---- scoring -------------------------------------------------------------------------------
    def _x(self, row: dict, k: str) -> float:
        return float(row.get(k, self.fill.get(k, 0.0)))

    def raw(self, row: dict) -> float:
        z = self.base
        for t in self.trees:
            node = t
            while "f" in node:
                node = node["l"] if self._x(row, node["f"]) <= node["t"] else node["r"]
            z += self.rate * node["v"]
        return z

    def predict(self, row: dict, group: str | None = None) -> float:
        return self._value(self.raw(row) + (self.offsets.get(group, 0.0) if group else 0.0))

    def contributions(self, row: dict) -> list[tuple[str, float]]:
        """Each feature's push on the raw score for this row, largest first (signed)."""
        c: dict[str, float] = {}
        for t in self.trees:
            node = t
            while "f" in node:
                nxt = node["l"] if self._x(row, node["f"]) <= node["t"] else node["r"]
                c[node["f"]] = c.get(node["f"], 0.0) + self.rate * (nxt["v"] - node["v"])
                node = nxt
        return sorted(c.items(), key=lambda kv: -abs(kv[1]))

    # ---- storage -------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"kind": self.loss, "names": self.names, "base": self.base, "rate": self.rate, "fill": self.fill,
                "trees": self.trees, "offsets": self.offsets}

    @classmethod
    def from_dict(cls, d: dict):
        m = cls(d.get("names") or [])
        m.base, m.rate, m.fill = float(d["base"]), float(d["rate"]), dict(d.get("fill") or {})
        m.trees, m.offsets = list(d["trees"]), dict(d.get("offsets") or {})
        return m


class BoostedRegressor(BoostedClassifier):
    """Boosted trees with squared loss. Output: the raw value (e.g. the log of a bill)."""

    loss = "squared"

    def _start(self, y: list) -> float:
        return sum(y) / len(y)

    def _grad(self, raw: float, y: float) -> tuple[float, float]:
        return raw - y, 1.0

    def _value(self, raw: float) -> float:
        return raw

    def fit_offsets(self, rows, y, groups, *, prior: float = 200.0) -> None:
        Linear.fit_offsets(self, rows, y, groups, prior=prior)


def _strip(node: dict) -> None:
    node.pop("j", None)
    node.pop("b", None)
    if "l" in node:
        _strip(node["l"])
        _strip(node["r"])
