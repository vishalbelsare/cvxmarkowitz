### Some temporary code until we decide on the final structure of the codebase

from collections import namedtuple

import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm


def mean_var_factor(
    n, k, sigma_tar, lower=-0.3, upper=0.4, leverage=2, lower_cash=None, upper_cash=None
):
    """
    Mean variance optimization with factor model
    """

    w = cp.Variable(n + 1, name="w")  # Last element is cash
    f = cp.Variable(k, name="f")

    w_old = cp.Parameter(n + 1, name="w_old")

    alpha_w = cp.Parameter(n, name="alpha_w")
    alpha_f = cp.Parameter(k, name="alpha_f")
    exposure = cp.Parameter((n, k), name="exposure")
    chol_f = cp.Parameter((k, k), name="chol_f")
    idio_vola = cp.Parameter(n, name="idio_vola")
    trading_costs = cp.Parameter(n, name="trading_costs")
    trading_costs_times_w_old = cp.Parameter(n, name="trading_costs_times_w_old")

    # if transaction_costs:
    ret = (
        alpha_f @ f
        + alpha_w @ w[:-1]
        - cp.norm1(cp.multiply(trading_costs, w[:-1]) - trading_costs_times_w_old)
    )
    # else:
    #     ret = alpha_f @ f + alpha_w @ w[:-1]

    obj = cp.Maximize(ret)

    risk = cp.norm2(
        cp.hstack([cp.norm2(chol_f.T @ f), cp.norm2(cp.multiply(idio_vola, w[:-1]))])
    )

    cons = [cp.sum(w) == 1, risk <= sigma_tar / np.sqrt(250), f == exposure.T @ w[:-1]]

    if lower is not None:
        cons.append(w[:-1] >= lower)
    if upper is not None:
        cons.append(w[:-1] <= upper)
    if leverage is not None:
        cons.append(cp.norm1(w[:-1]) <= leverage)

    if lower_cash is not None:
        cons.append(w[-1] >= lower_cash)
    if upper_cash is not None:
        cons.append(w[-1] <= upper_cash)

    prob = cp.Problem(obj, cons)

    return (
        prob,
        w,
        f,
        alpha_w,
        alpha_f,
        exposure,
        chol_f,
        idio_vola,
        trading_costs,
        trading_costs_times_w_old,
        w_old,
    )


def backtest_markowitz_factor(
    returns,
    alphas_w,
    alphas_f,
    exposures,
    Sigmas_f,
    idio_volas,
    sigma_tar,
    lower=-0.3,
    upper=0.4,
    leverage=2,
    lower_cash=None,
    upper_cash=None,
    transaction_cost=1,
    metrics_transaction_cost=1,
):
    """
    param returns: DataFrame with realized returns
    param alphas_w: DataFrame with predicted (excess,\
          after accounting for factors) asset returns (causal)
    param alphas_f: DataFrame with predicted factor returns (causal)
    param Sigmas_f: Dict of factor covariance matrices
    param exposures: Dict with factor exposures (factor loading matrices)
    param sigma_tar: Target volatility
    param lower: Lower bound on weights
    param upper: Upper bound on weights
    param leverage: Leverage bound
    param lower_cash: Lower bound on cash
    param upper_cash: Upper bound on cash
    param transaction_cost: Transaction cost (is multiplied by 2 BPS)
    param metrics_transaction_cost: Transaction cost used for computing metrics
    (is multiplied by 2 BPS)

    returns: DataFrame with portfolio weights; portfolio metrics
    """
    n = alphas_w.shape[1]
    k = alphas_f.shape[1]

    (
        prob,
        w,
        f,
        alpha_w,
        alpha_f,
        exposure,
        chol_f,
        idio_vola,
        trading_costs,
        trading_costs_times_w_old,
        w_old,
    ) = mean_var_factor(
        n,
        k,
        sigma_tar,
        lower=lower,
        upper=upper,
        leverage=leverage,
        lower_cash=lower_cash,
        upper_cash=upper_cash,
    )

    porfolio_weights = {}
    times = [*Sigmas_f.keys()]

    # Start with cash only
    w_old.value = np.zeros(returns.shape[1] + 1)
    w_old.value[-1] = 1

    # for time in tqdm(times):
    for time in times:
        if time not in alphas_w.index or time not in alphas_f.index:
            continue

        # Get new data
        alpha_w_hat = alphas_w.loc[time]
        alpha_f_hat = alphas_f.loc[time]
        exposure_hat = exposures[time]
        Sigma_f_hat = Sigmas_f[time]
        idio_vola_hat = idio_volas.loc[time]

        trading_costs_hat = (
            np.ones(returns.shape[1]) * 2 * (0.01) ** 2 * transaction_cost
        )
        trading_costs_times_w_old_hat = trading_costs_hat * w_old.value[:-1]

        # Update problem data
        alpha_w.value = alpha_w_hat.values
        alpha_f.value = alpha_f_hat.values
        exposure.value = exposure_hat.values
        if type(Sigma_f_hat) == pd.DataFrame:
            chol_f.value = np.linalg.cholesky(Sigma_f_hat.values)
        else:
            chol_f.value = np.linalg.cholesky(Sigma_f_hat.reshape((1, 1)))
        idio_vola.value = idio_vola_hat.values
        trading_costs.value = trading_costs_hat
        trading_costs_times_w_old.value = trading_costs_times_w_old_hat

        # Solve problem
        prob.solve(solver="CLARABEL", ignore_dpp=False)

        # Get new weights
        weights = pd.concat(
            [
                pd.Series(w.value[:-1], index=returns.columns),
                pd.Series(w.value[-1], index=["cash"]),
            ]
        )
        porfolio_weights[time] = weights

        # Update old weights
        w_old.value = w.value

    porfolio_weights = pd.DataFrame(porfolio_weights).T
    metrics = portfolio_metrics(
        returns, porfolio_weights, metrics_transaction_cost * (0.01) ** 2 * 2
    )

    return porfolio_weights, metrics


def max_sharpe(n, lower=-0.3, upper=0.4, leverage=2):
    """
    Returns parameterized max sharpe ratio optimization problem
    """

    z = cp.Variable(n, name="z")
    alpha = cp.Parameter(n, name="alpha")
    chol = cp.Parameter((n, n), name="chol")

    z_risk = cp.norm2(chol.T @ z)

    cons = [alpha @ z == 1]
    if lower is not None:
        cons.append(z >= lower * cp.sum(z))
    if upper is not None:
        cons.append(z <= upper * cp.sum(z))
    if leverage is not None:
        cons.append(cp.norm1(z) <= leverage * cp.sum(z))

    obj = cp.Minimize(z_risk)
    prob = cp.Problem(obj, cons)

    return prob, z, alpha, chol


def backtest_max_sharpe(
    returns,
    alphas,
    Sigmas,
    lower=-0.3,
    upper=0.4,
    leverage=2,
    metrics_transaction_cost=1,
):
    """
    param returns: DataFrame with realized returns
    param alphas: DataFrame with predicted returns (causal)
    param Sigmas: Dict with covariance matrices
    param lower: Lower bound on weights
    param upper: Upper bound on weights
    param leverage: Leverage bound
    param metrics_transaction_cost: Transaction cost used for computing metrics
    (is multiplied by 2 BPS)

    returns: DataFrame with portfolio weights; portfolio metrics
    """

    prob, z, alpha, chol = max_sharpe(
        returns.shape[1], lower=lower, upper=upper, leverage=leverage
    )

    porfolio_weights = {}
    times = [*Sigmas.keys()]

    for time in times:
        if time not in alphas.index:
            continue

        # Get new data
        alpha_hat = alphas.loc[time]
        Sigma_hat = Sigmas[time]

        # Update problem data
        alpha.value = alpha_hat.values
        if type(Sigma_hat) == pd.DataFrame:
            chol.value = np.linalg.cholesky(Sigma_hat.values)
        else:
            chol.value = np.linalg.cholesky(Sigma_hat.reshape((1, 1)))

        # Solve problem
        prob.solve(solver="CLARABEL", ignore_dpp=False)
        if prob.status == "infeasible":
            print(alpha_hat)
            print(Sigma_hat)

        # Get new weights
        w = z.value / np.sum(z.value)
        weights = pd.Series(w, index=returns.columns)
        porfolio_weights[time] = weights

    porfolio_weights = pd.DataFrame(porfolio_weights).T
    metrics = portfolio_metrics(
        returns, porfolio_weights, metrics_transaction_cost * (0.01) ** 2 * 2
    )

    return porfolio_weights, metrics


def mean_variance(
    n,
    sigma_tar,
    lower=-0.3,
    upper=0.4,
    leverage=2,
    lower_cash=None,
    upper_cash=None,
    transaction_costs=False,
):
    """
    Returns parameterized mean variance optimization problem
    """

    w = cp.Variable(n + 1, name="w")  # Last element is cash
    w_old = cp.Parameter(n + 1, name="w_old")

    alpha = cp.Parameter(n, name="alpha")
    alpha_uncertainty = cp.Parameter(n, nonneg=True, name="alpha_uncertainty")
    chol = cp.Parameter((n, n), name="chol")

    # trading_costs_times_w_old to make it DPP
    trading_costs = cp.Parameter(n, nonneg=True, name="trading_costs")
    trading_costs_times_w_old = cp.Parameter(n, name="trading_costs_times_w_old")

    if transaction_costs:
        ret = (
            alpha @ w[:-1]
            - alpha_uncertainty @ cp.abs(w[:-1])
            - cp.norm1(cp.multiply(trading_costs, w[:-1]) - trading_costs_times_w_old)
        )
    else:
        ret = alpha @ w[:-1] - alpha_uncertainty @ cp.abs(w[:-1])

    obj = cp.Maximize(ret)

    risk = cp.norm2(chol.T @ w[:-1])

    cons = [cp.sum(w) == 1, risk <= sigma_tar / np.sqrt(250)]
    if lower is not None:
        cons.append(w[:-1] >= lower)
    if upper is not None:
        cons.append(w[:-1] <= upper)
    if leverage is not None:
        cons.append(cp.norm1(w[:-1]) <= leverage)

    if lower_cash is not None:
        cons.append(w[-1] >= lower_cash)
    if upper_cash is not None:
        cons.append(w[-1] <= upper_cash)

    prob = cp.Problem(obj, cons)

    return (
        prob,
        w,
        alpha,
        alpha_uncertainty,
        chol,
        w_old,
        trading_costs,
        trading_costs_times_w_old,
    )


def min_risk(n, lower=-0.3, upper=0.4, leverage=2):
    """
    Returns parameterized minimum variance optimization problem
    """

    w = cp.Variable(n)

    chol = cp.Parameter((n, n))

    risk = cp.norm2(chol.T @ w)

    obj = cp.Minimize(risk)
    cons = [cp.sum(w) == 1, w >= lower, w <= upper, cp.norm1(w) <= leverage]

    prob = cp.Problem(obj, cons)

    return prob, w, chol


Metrics = namedtuple(
    "Metrics",
    [
        "daily_returns",
        "daily_value",
        "mean_return",
        "std_return",
        "sharpe_ratio",
        "drawdown",
        "turnover",
    ],
)


def drawdown(portfolio_value):
    peak = portfolio_value[0]  # Tracks the peak value encountered so far
    drawdowns = [0]  # List to store drawdown values

    for value in portfolio_value[1:]:
        peak = max(peak, value)
        drawdown = (peak - value) / peak
        drawdowns.append(drawdown)

    return pd.Series(drawdowns, index=portfolio_value.index)


def dilute_with_cash(weights, Sigma, sigma_tar):
    """
    Dilutes weights with cash to achieve target volatility
    """

    sigma = np.sqrt(weights.T @ Sigma @ weights) * np.sqrt(250)

    theta = sigma_tar / sigma

    # New weights is theta * weights concatenated with 1 - theta

    weights_diluted = pd.concat([theta * weights, pd.Series(1 - theta, index=["cash"])])

    return weights_diluted


def get_turnover(weights):
    """
    Computes the average (annualized) turnover of a portfolio
    """

    w_prev = weights.iloc[0]
    turnover = 0
    for _, w in weights.iloc[1:].iterrows():
        turnover += np.sum(np.abs(w[:-1] - w_prev[:-1])) / np.sum(
            np.abs(w_prev[:-1])
        )  # Don't count cash
        w_prev = w

    return turnover / (weights.shape[0] - 1) * 250


def portfolio_metrics(returns, weights, trading_costs=(0.01) ** 2 * 2, spreads=None):
    """
    Computes portfolio growth given realized returns and portfolio weights
    """

    assert set(weights.index).issubset(set(returns.index))
    "Weights index must be a subset of returns index"

    returns_relevant = returns.loc[weights.index]

    portfolio_returns = (returns_relevant * weights).sum(axis=1)

    if trading_costs is not None:
        trades = (weights.shift(1) - weights).abs()
        trades.iloc[:, -1] = 0  # No trading costs on cash
        if trading_costs is not None:
            portfolio_returns -= (trades * trading_costs).sum(axis=1)
        elif spreads is not None:
            trading_costs = spreads / 2
            portfolio_returns -= (trades * trading_costs).sum(axis=1)

    portfolio_value = (1 + portfolio_returns).cumprod()

    mean_return = portfolio_returns.mean() * 250
    std_return = portfolio_returns.std() * np.sqrt(250)
    sharpe_ratio = mean_return / std_return

    drawdowns = drawdown(portfolio_value)

    turnover = get_turnover(weights)

    return Metrics(
        portfolio_returns,
        portfolio_value,
        mean_return,
        std_return,
        sharpe_ratio,
        drawdowns,
        turnover,
    )


def backtest_markowitz(
    returns,
    alphas,
    Sigmas,
    sigma_tar,
    lower=-0.3,
    upper=0.4,
    leverage=2,
    lower_cash=None,
    upper_cash=None,
    transaction_cost=1,
    metrics_transaction_cost=1,
    alpha_uncertainties=None,
    spreads=None,
):
    """
    param returns: DataFrame with realized returns
    param alphas: DataFrame with predicted returns (causal)
    param Sigmas: Dict with covariance matrices
    param sigma_tar: Target volatility
    param lower: Lower bound on weights
    param upper: Upper bound on weights
    param leverage: Leverage bound
    param lower_cash: Lower bound on cash
    param upper_cash: Upper bound on cash
    param transaction_cost: Transaction cost (is multiplied by 2 BPS)
    param metrics_transaction_cost: Transaction cost used for computing metrics
    (is multiplied by 2 BPS)

    returns: DataFrame with portfolio weights; portfolio metrics
    """

    (
        prob,
        w,
        alpha,
        alpha_uncertainty,
        chol,
        w_old,
        trading_costs,
        trading_costs_times_w_old,
    ) = mean_variance(
        returns.shape[1],
        sigma_tar,
        lower=lower,
        upper=upper,
        leverage=leverage,
        lower_cash=lower_cash,
        upper_cash=upper_cash,
        transaction_costs=True,
    )

    porfolio_weights = {}
    times = [*Sigmas.keys()]

    # Start with cash only
    w_old.value = np.zeros(returns.shape[1] + 1)
    w_old.value[-1] = 1

    if alpha_uncertainties is None:
        alpha_uncertainties = alphas * 0

    for time in tqdm(times):
        if time not in alphas.index:
            continue

        # Get new data
        alpha_hat = alphas.loc[time]
        # alpha_uncertainty_hat = pd.Series(np.zeros(returns.shape[1]))
        alpha_uncertainty_hat = alpha_uncertainties.loc[time]
        Sigma_hat = Sigmas[time]
        trading_costs_hat = (
            np.ones(returns.shape[1]) * 2 * (0.01) ** 2 * transaction_cost
        )
        trading_costs_times_w_old_hat = trading_costs_hat * w_old.value[:-1]

        # Update problem data
        alpha.value = alpha_hat.values
        alpha_uncertainty.value = alpha_uncertainty_hat.values
        if type(Sigma_hat) == pd.DataFrame:
            chol.value = np.linalg.cholesky(Sigma_hat.values)
        else:
            chol.value = np.linalg.cholesky(Sigma_hat.reshape((1, 1)))
        trading_costs.value = trading_costs_hat
        trading_costs_times_w_old.value = trading_costs_times_w_old_hat

        # Solve problem
        prob.solve(solver="CLARABEL", ignore_dpp=False)
        # print(prob.status)

        # Get new weights
        weights = pd.concat(
            [
                pd.Series(w.value[:-1], index=returns.columns),
                pd.Series(w.value[-1], index=["cash"]),
            ]
        )
        porfolio_weights[time] = weights

        # Update old weights
        w_old.value = w.value

    porfolio_weights = pd.DataFrame(porfolio_weights).T
    if metrics_transaction_cost is not None:
        metrics = portfolio_metrics(
            returns,
            porfolio_weights,
            metrics_transaction_cost * (0.01) ** 2 * 2,
            spreads=None,
        )
    elif spreads is not None:
        metrics = portfolio_metrics(
            returns, porfolio_weights, metrics_transaction_cost=0, spreads=spreads
        )

    return porfolio_weights, metrics


def single_asset_markowitz_l2(
    returns, alphas, sigmas, gamma, transaction_cost, gamma_tr, simulation_cost=True
):
    """
    param alphas: ps.Series of alphas (causal return predictions)
    param sigmas: pd.Series of sigmas (causal volatility predictions)
    """
    theta_old = 0

    assert alphas.index.equals(sigmas.index)

    thetas = pd.Series(index=alphas.index)
    for time in alphas.index:
        alpha = alphas.loc[time]
        sigma = sigmas.loc[time]

        theta_new = (alpha + 2 * gamma_tr * transaction_cost * theta_old) / (
            2 * (gamma * sigma**2 + gamma_tr * transaction_cost)
        )

        thetas.loc[time] = theta_new
        theta_old = theta_new

    # Compute realized portfolio returns
    portfolio_returns = pd.Series(index=alphas.index)
    for i in range(len(alphas.index)):
        time = alphas.index[i]
        if i == 0:
            portfolio_returns.loc[time] = (
                thetas.loc[time] * returns.loc[time]
                - transaction_cost * np.abs(thetas.loc[time]) * simulation_cost
            )
        else:
            time_prev = alphas.index[i - 1]
            portfolio_returns.loc[time] = (
                thetas.loc[time] * returns.loc[time]
                - transaction_cost
                * np.abs(thetas.loc[time] - thetas.loc[time_prev])
                * simulation_cost
            )


def single_asset_markowitz_l1_old(
    returns, alphas, sigmas, gamma, transaction_cost, gamma_tr, simulation_cost=True
):
    """
    param alphas: ps.Series of alphas (causal return predictions)
    param sigmas: pd.Series of sigmas (causal volatility predictions)
    """
    theta_old = 0

    assert alphas.index.equals(sigmas.index)

    thetas = pd.Series(index=alphas.index)
    for time in alphas.index:
        alpha = alphas.loc[time]
        sigma = sigmas.loc[time]

        if theta_old < (alpha - gamma_tr * transaction_cost) / (2 * gamma * sigma**2):
            theta_new = (alpha - gamma_tr * transaction_cost) / (2 * gamma * sigma**2)
        elif theta_old > (alpha + gamma_tr * transaction_cost) / (
            2 * gamma * sigma**2
        ):
            theta_new = (alpha + gamma_tr * transaction_cost) / (2 * gamma * sigma**2)
        else:
            theta_new = theta_old

        thetas.loc[time] = theta_new
        theta_old = theta_new

    # Compute realized portfolio returns
    portfolio_returns = pd.Series(index=alphas.index)
    for i in range(len(alphas.index)):
        time = alphas.index[i]
        if i == 0:
            portfolio_returns.loc[time] = (
                thetas.loc[time] * returns.loc[time]
                - transaction_cost * np.abs(thetas.loc[time]) * simulation_cost
            )
        else:
            time_prev = alphas.index[i - 1]
            portfolio_returns.loc[time] = (
                thetas.loc[time] * returns.loc[time]
                - transaction_cost
                * np.abs(thetas.loc[time] - thetas.loc[time_prev])
                * simulation_cost
            )

    return thetas, portfolio_returns


def single_asset_markowitz_l1(
    returns, alphas, sigmas, sigma_tar, transaction_cost, gamma_tr, simulation_cost=True
):
    """
    Analytic Markowitz with linear trading costs

    param alphas: ps.Series of alphas (causal return predictions)
    param sigmas: pd.Series of sigmas (causal volatility predictions)
    """
    theta_old = 0

    sigma_tar = sigma_tar / np.sqrt(250)

    assert alphas.index.equals(sigmas.index)

    thetas = pd.Series(index=alphas.index)
    for time in alphas.index:
        alpha = alphas.loc[time]
        sigma = sigmas.loc[time]

        omega = gamma_tr * transaction_cost

        # Case 3: theta = theta_old
        if np.abs(alpha) <= omega and theta_old <= sigma_tar / sigma:
            theta_new = theta_old
        elif (
            (alpha + omega) / (2 * sigma * sigma_tar) >= 0
            or (alpha - omega) / (2 * sigma * sigma_tar) <= 0
        ) and theta_old == sigma_tar / sigma:
            theta_new = theta_old
        # Case 1: theta > theta_old
        elif alpha >= omega and theta_old < sigma_tar / sigma:
            theta_new = sigma_tar / sigma
        elif alpha <= omega and theta_old < -sigma_tar / sigma:
            theta_new = -sigma_tar / sigma
        # Case 2: theta < theta_old
        elif alpha >= -omega and theta_old > sigma_tar / sigma:
            theta_new = sigma_tar / sigma
        elif alpha <= -omega and theta_old > -sigma_tar / sigma:
            theta_new = -sigma_tar / sigma

        thetas.loc[time] = theta_new
        theta_old = theta_new

    # Compute realized portfolio returns
    portfolio_returns = pd.Series(index=alphas.index)
    for i in range(len(alphas.index)):
        time = alphas.index[i]
        if i == 0:
            portfolio_returns.loc[time] = (
                thetas.loc[time] * returns.loc[time]
                - transaction_cost * np.abs(thetas.loc[time]) * simulation_cost
            )
        else:
            time_prev = alphas.index[i - 1]
            portfolio_returns.loc[time] = (
                thetas.loc[time] * returns.loc[time]
                - transaction_cost
                * np.abs(thetas.loc[time] - thetas.loc[time_prev])
                * simulation_cost
            )

    return thetas, portfolio_returns


def single_asset_markowitz_cvx(
    returns,
    alphas,
    sigmas,
    gamma,
    transaction_cost,
    gamma_tr,
    tr_cost="lin",
    simulation_cost=True,
):
    """
    Single asset Markowitz using cvxpy
    """

    assert alphas.index.equals(sigmas.index)

    theta = cp.Variable(name="theta")

    theta_old = cp.Parameter()
    r = cp.Parameter()
    var = cp.Parameter(nonneg=True)

    cons = [var * cp.square(theta) <= 0.1**2 / 250]
    gamma = 0
    if tr_cost == "lin":
        obj = cp.Minimize(
            gamma * var * cp.square(theta)
            + gamma_tr * transaction_cost * cp.abs(theta - theta_old)
            - r * theta
        )
    elif tr_cost == "quad":
        obj = cp.Minimize(
            gamma * var * cp.square(theta)
            + gamma_tr * transaction_cost * (theta - theta_old) ** 2
            - r * theta
        )

    prob = cp.Problem(obj, cons)
    # prob = cp.Problem(obj)

    thetas = pd.Series(index=alphas.index)
    theta.value = 0
    for time in alphas.index:
        alpha = alphas.loc[time]
        sigma = sigmas.loc[time]

        theta_old.value = theta.value
        r.value = alpha
        var.value = sigma**2

        prob.solve(solver="CLARABEL", ignore_dpp=False)

        thetas.loc[time] = theta.value

    # Compute realized portfolio returns
    portfolio_returns = pd.Series(index=alphas.index)
    for i in range(len(alphas.index)):
        time = alphas.index[i]
        if i == 0:
            portfolio_returns.loc[time] = (
                thetas.loc[time] * returns.loc[time]
                - transaction_cost * np.abs(thetas.loc[time]) * simulation_cost
            )
        else:
            time_prev = alphas.index[i - 1]
            portfolio_returns.loc[time] = (
                thetas.loc[time] * returns.loc[time]
                - transaction_cost
                * np.abs(thetas.loc[time] - thetas.loc[time_prev])
                * simulation_cost
            )

    return thetas, portfolio_returns
