# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.markowitz.builder import Builder
from cvx.markowitz.risk import SampleCovariance


@dataclass
class DummyBuilder(Builder):
    @property
    def objective(self):
        return cp.Maximize(0.0)


def test_dummy():
    solver = DummyBuilder(assets=1)
    solver.model = {"risk": SampleCovariance(assets=1)}
    solver.variables = {"weights": cp.Variable(1)}

    solver.update(cov=np.eye(1))
    problem = solver.build()
    problem.solve()

    solver.variables["weights"].value = np.array([2.0])
    d = solver.solution(names=["a"])
    assert d["a"] == 2.0