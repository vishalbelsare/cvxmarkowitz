# -*- coding: utf-8 -*-
"""Bounds"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .model import Model


@dataclass
class Bounds(Model):
    m: int = 0
    name: str = ""

    def estimate(self, weights, **kwargs):
        """No estimation for bounds"""
        raise NotImplementedError("No estimation for bounds")

    def _f(self, str):
        return f"{str}_{self.name}"

    def __post_init__(self):
        self.data[self._f("lower")] = cp.Parameter(
            shape=self.m,
            name="lower bound",
            value=np.zeros(self.m),
        )
        self.data[self._f("upper")] = cp.Parameter(
            shape=self.m,
            name="upper bound",
            value=np.ones(self.m),
        )

    def update(self, **kwargs):
        lower = kwargs[self._f("lower")]
        self.data[self._f("lower")].value = np.zeros(self.m)
        self.data[self._f("lower")].value[: len(lower)] = lower

        upper = kwargs[self._f("upper")]  # .get("upper", np.ones(self.m))
        self.data[self._f("upper")].value = np.zeros(self.m)
        self.data[self._f("upper")].value[: len(upper)] = upper

    def constraints(self, weights, **kwargs):
        return {
            f"lower bound {self.name}": weights >= self.data[self._f("lower")],
            f"upper bound {self.name}": weights <= self.data[self._f("upper")],
        }

    @property
    def assets(self):
        return self.m
