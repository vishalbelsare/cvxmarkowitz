# -*- coding: utf-8 -*-
import mosek.fusion as fusion
import pandas as pd
from loguru import logger

from cvx.linalg.cholesky import cholesky
from cvx.simulator.builder import builder

if __name__ == "__main__":
    prices = pd.read_csv(
        "data/stock_prices.csv", index_col=0, header=0, parse_dates=True
    )

    # --------------------------------------------------------------------------------------------
    # construct the portfolio using a builder
    b = builder(prices=prices)

    # --------------------------------------------------------------------------------------------
    # compute data needed for the portfolio construction
    cov = dict(b.cov(halflife=10, min_periods=30))

    for t, _ in b:
        try:
            with fusion.Model("minVar") as M:
                weights = M.variable("weights", 20, fusion.Domain.inRange(0.0, 1.0))
                _z = M.variable("z", 20)
                risk = M.variable("risk", 1, fusion.Domain.greaterThan(0.0))

                M.constraint(
                    "fully-invested",
                    fusion.Expr.sum(weights),
                    fusion.Domain.equalsTo(1.0),
                )
                M.constraint(
                    "risk-coordinate",
                    fusion.Expr.sub(
                        fusion.Expr.mul(cholesky(cov[t[-1]].values), weights), _z
                    ),
                    fusion.Domain.equalsTo(0.0),
                )

                # Create the aliases
                z1 = fusion.Var.vstack(risk, _z)
                qc1 = M.constraint("qc1", z1, fusion.Domain.inQCone())
                logger.debug(t[-1])

                M.objective("obj", fusion.ObjectiveSense.Minimize, risk)

                # Solve the problem
                M.solve()

                # Get the solution values
                sol = weights.level()
                www = pd.Series(index=prices.columns, data=sol)

                # update the builder
                b.set_weights(t[-1], weights=www)
        except KeyError:
            pass

    # --------------------------------------------------------------------------------------------
    # build the portfolio
    portfolio = b.build()
    portfolio.snapshot()
    portfolio.metrics()
