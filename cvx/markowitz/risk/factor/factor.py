# -*- coding: utf-8 -*-
"""Factor risk model
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.linalg import cholesky
from cvx.markowitz import Model
from cvx.markowitz.bounds import Bounds


@dataclass
class FactorModel(Model):
    """Factor risk model"""

    assets: int = 0
    k: int = 0

    def __post_init__(self):
        self.data["exposure"] = cp.Parameter(
            shape=(self.k, self.assets),
            name="exposure",
            value=np.zeros((self.k, self.assets)),
        )

        self.data["idiosyncratic_risk"] = cp.Parameter(
            shape=self.assets, name="idiosyncratic risk", value=np.zeros(self.assets)
        )

        self.data["chol"] = cp.Parameter(
            shape=(self.k, self.k),
            name="cholesky of covariance",
            value=np.zeros((self.k, self.k)),
        )

        self.bounds_assets = Bounds(assets=self.assets, name="assets")
        self.bounds_factors = Bounds(assets=self.k, name="factors")

    def estimate(self, weights, **kwargs):
        """
        Compute the total variance
        """
        var_residual = cp.norm2(cp.multiply(self.data["idiosyncratic_risk"], weights))

        y = kwargs.get("factor_weights", self.data["exposure"] @ weights)

        return cp.norm2(cp.vstack([cp.norm2(self.data["chol"] @ y), var_residual]))

    def update(self, **kwargs):
        exposure = kwargs["exposure"]
        k, assets = exposure.shape

        self.data["exposure"].value[:k, :assets] = kwargs["exposure"]
        self.data["idiosyncratic_risk"].value[:assets] = kwargs["idiosyncratic_risk"]
        self.data["chol"].value[:k, :k] = cholesky(kwargs["cov"])
        self.bounds_assets.update(**kwargs)
        self.bounds_factors.update(**kwargs)

    def constraints(self, weights, **kwargs):
        y = kwargs.get("factor_weights", self.data["exposure"] @ weights)

        factor = {"factors": y == self.data["exposure"] @ weights}

        return (
            self.bounds_assets.constraints(weights)
            | self.bounds_factors.constraints(y)
            | factor
        )

    @property
    def variables(self):
        return cp.Variable(self.assets), cp.Variable(self.k)