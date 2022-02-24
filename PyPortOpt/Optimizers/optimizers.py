"""
The PortOpt application is powered by multiple optimizers designed to implement theory in an elegant 
and easy to use way.

This module consists all the functions required to run a portfolio optimization using parameters 
that the user inputs
"""
import logging
import math
import numpy as np
from numpy import linalg as LA
import osqp
import pandas as pd
from scipy import sparse
from scipy.stats import norm
import logging
from .G_functions import *
from ._utils import (
    _create_wealth_grid,
    _create_weight_function,
    _expected_value
)
from .reinforce import update_policy_path


##################
#  LOGGING SETUP #
##################
debug = False

if debug:
    logMode = logging.DEBUG
else:
    logMode = logging.INFO

# create logger
logger = logging.getLogger("optimizers")
# set log level for all handlers to debug
logger.setLevel(logMode)

# create console handler and set level to debug
# best for development or debugging
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logMode)

# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# add formatter to ch
consoleHandler.setFormatter(formatter)

# add ch to logger
logger.addHandler(consoleHandler)
######################
#  END LOGGING SETUP #
######################


def testFunction():
    """
    Function to test if the import is working

    Parameters
    ----------
        This function has no parameters

    Returns
    ----------
        This function returns true
    """
    return True


def preprocessData(data):
    """
    Helper function to create a covariance matrix and mean vector

    Parameters
    ----------
    data : Dictionary
        Dictionary containing Date, Ticker and Adjusted Close price

    Returns
    -------
    meanVec : Vector
    sigMat : Matrix
    """

    if isinstance(data, dict):
        data = pd.DataFrame.from_dict(data)
        df = data[["Date", "Ticker", "Adjusted_Close"]]
        df.columns = ["date", "ticker", "price"]
        df1 = df.pivot_table(index="date", columns="ticker", values="price")

    elif isinstance(data, pd.DataFrame):
        df1 = data.dropna(axis=1)
    df_logret = 100 * np.log(df1).diff().dropna()
    logret = df_logret.values

    sigMat = np.cov(logret, rowvar=False)
    meanVec = np.mean(logret, axis=0)
    return meanVec, sigMat, df_logret


def SymPDcovmatrix(A, tol=None):
    """
    function corrects a covariance matrix A to be symmetric positive definite
    it uses eigenvalue decomposition and shifts all small eigenvalues to tol

    Parameters
    ----------
    A : Array like object
    tol : float
        (optional, default tol = 1e-04) minimum value for all eigenvalues

    Returns
    -------
    A : Array
        corrected matrix A.
    e_min : float
        minimum value for all eigenvalues
    """
    m, n = A.shape
    if n != m:
        print("Input matrix has to be a square matrix ")
    if not tol:
        tol = 1e-04
    A = (A + A.transpose()) / 2
    D, V = LA.eig(A)
    for i in range(len(D)):
        if D[i] < tol:
            D[i] = tol

    D = np.diag(D)
    t = np.dot(V, D)
    A = np.dot(t, V.transpose())
    e_min = max(tol, min(np.diag(D)))
    A = (A + A.transpose()) / 2
    return A, e_min


def sigMatShrinkage(sigMat, lambda_l2):
    """
    Function to shrink the covariance matrix

    Parameters
    ----------
    sigMat : Matrix
    lambda_l2 : Float

    Returns
    -------
    D : Array
    """
    d = sigMat.shape[0]
    sig = np.sqrt(np.diag(sigMat))
    t = np.dot(np.diag(1 / sig), sigMat)
    corrMat = np.dot(t, np.diag(1 / sig))
    corrs = None
    for k in range(d - 1):
        if corrs is None:
            corrs = np.diag(corrMat, k + 1)
        else:
            corrs = np.hstack([corrs, np.diag(corrMat, k + 1)])
    if 1 == 1:
        sigMat = sigMat + lambda_l2 * np.mean(sig ** 2) * np.eye(d)
    else:
        t = np.dot(
            np.mean(sig) * np.eye(d),
            np.eye(d) + (np.ones((d, d)) - np.eye(d)) * np.mean(corrs),
        )
        sigMat = sigMat * (1 - lambda_l2) + lambda_l2 * np.dot(
            t, np.mean(sig) * np.eye(d)
        )
    return sigMat


def Dmat(n, k):
    """
    function reform a matrix for assets with order

    Parameters
    ----------
    n : int
    k : int

    Returns
    -------
    D : Array
    """
    if k == 0:
        D = np.eye(n)
    elif k == 1:
        D = np.eye(n - 1, n)
        for i in range(n - 1):
            D[i, i + 1] = -1
    else:
        D = Dmat(n, 1)
        for i in range(k - 1):
            Dn = Dmat(n - i - 1, 1)
            D = np.dot(Dn, D)
    return D


def minimumVariancePortfolio(
    sigMat, longShort, maxAlloc=1, lambda_l1=0, lambda_l2=0, assetsOrder=None
):
    """
    Optimizes portfolio for minimum variance

    Parameters
    ----------
    SigMat : Matrix
    LongShort : Float
        Takes value between 0 and 1
    maxAlloc : Float
        Takes value between 0 and 1. Specifies the maximum weight an asset can get
    lambda_l1 : Float
        Takes a value greater than 0. Specifies L1 penalty
    lambda_l2 : Float
        Takes a value greater than 0. Specifies L2 penalty

    Returns
    -------
    w_opt : Array
        Returns the weights of given to each asset in form of a numpy array
    var_opt : Float
        Returns the variance of the portfolio
    """
    d = sigMat.shape[0]

    if assetsOrder:
        temp = sigMat[:, assetsOrder]
        sigMat = temp[assetsOrder, :]
    if lambda_l2:
        sigMat_l2 = sigMatShrinkage(sigMat, lambda_l2)
        sigMat_l2, e_min = SymPDcovmatrix(sigMat_l2)
    else:
        sigMat, e_min = SymPDcovmatrix(sigMat)

    if longShort == 0:
        Aeq = np.ones(d)
        Beq = 1
        LB = np.zeros(d)
        UB = maxAlloc * np.ones(d)
        if assetsOrder:
            L_ine = -np.ones(d - 1)
            D = np.eye(d - 1, d)
            for i in range(d - 1):
                D[i, i + 1] = -1
            A = -1 * D
            B = np.zeros(d - 1)
            A = np.vstack([A, Aeq, np.eye(d)])
            l = np.hstack([L_ine, Beq, LB])
            u = np.hstack([B, Beq, UB])
        else:
            A = np.vstack([Aeq, np.eye(d)])
            l = np.hstack([Beq, LB])
            u = np.hstack([Beq, UB])

        if lambda_l1:
            meanVec = -lambda_l1 * np.ones(d)
        else:
            meanVec = -np.zeros(d)

        if lambda_l2:
            P = sparse.csc_matrix(sigMat_l2)
        else:
            P = sparse.csc_matrix(sigMat)

        A = sparse.csc_matrix(A)

        prob = osqp.OSQP()
        # Setup workspace
        prob.setup(P, -meanVec, A, l, u, verbose=False, max_iter = 10000, eps_abs=1e-8, eps_rel = 1e-8,eps_prim_inf = 1e-8,eps_dual_inf = 1e-8)
        # Solve problem
        res = prob.solve()
        w_opt = res.x
        if not w_opt.all():
            w_opt = np.ones(d) / d

    elif longShort != 0:
        A = np.hstack([np.zeros(d), np.ones(d), np.zeros(d)])
        B = 1 + abs(longShort)
        Grenze = min(abs(longShort), maxAlloc)
        if assetsOrder:
            L_ine = np.hstack([0, -(1 + 2 * Grenze) * np.ones(d - 1)])
            D = np.eye(d - 1, d)
            for i in range(d - 1):
                D[i, i + 1] = -1
            A = np.vstack([A, np.hstack([-1 * D, np.zeros((d - 1, 2 * d))])])
            B = np.hstack([B, np.zeros(d - 1)])
        else:
            L_ine = 0
        Aeq = np.vstack(
            [
                np.hstack([np.eye(d), -np.eye(d), np.eye(d)]),
                np.hstack([np.ones(d), np.zeros(d), np.zeros(d)]),
            ]
        )
        Beq = np.hstack([np.zeros(d), 1])
        LB = np.hstack([-Grenze * np.ones(d), np.zeros(2 * d)])
        UB = maxAlloc * np.ones(3 * d)
        if lambda_l2:
            sigMat3d = np.vstack([
                np.hstack([sigMat_l2, np.zeros((d, 2 * d))]),
                np.zeros((2 * d, 3 * d))
            ])
        else:
            sigMat3d = np.vstack([
                np.hstack([sigMat, np.zeros((d, 2 * d))]),
                np.zeros((2 * d, 3 * d))
            ])

        sigMat3d = sigMat3d + np.diag(
            np.hstack([-0.1 * e_min * np.ones(d), 0.1 * e_min * np.ones(2 * d)])
        )

        if lambda_l1:
            meanvec3d = np.hstack([np.zeros(d), -lambda_l1 * np.ones(2 * d)])
        else:
            meanvec3d = np.hstack([np.zeros(d), np.zeros(2 * d)])

        A = np.vstack([A, Aeq, np.eye(3 * d)])
        l = np.hstack([L_ine, Beq, LB])
        u = np.hstack([B, Beq, UB])

        A = sparse.csc_matrix(A)
        sigMat3d = sparse.csc_matrix(sigMat3d)

        prob = osqp.OSQP()
        # Setup workspace
        prob.setup(sigMat3d, -meanvec3d, A, l, u, verbose=False, eps_abs=1e-8, eps_rel = 1e-8,eps_prim_inf = 1e-8,eps_dual_inf = 1e-8)
        # Solve problem
        res = prob.solve()
        wuv_opt = res.x
        if not wuv_opt.all():
            w_opt = np.ones(d) / d
        else:
            w_opt = wuv_opt[:d]

    t = np.dot(w_opt, sigMat)
    Var_opt = np.dot(t, w_opt.transpose())
    if assetsOrder:
        w_opt = w_opt[assetsOrder]
    # if exitflag!=1:
    # print("minimumVariancePortfolio: Exitflag different than 1 in quadprog")
    return w_opt, Var_opt


def meanVariancePortfolioReturnsTarget(
        meanVec,
        sigMat,
        retTarget,
        longShort,
        maxAlloc=1,
        lambda_l1=0,
        lambda_l2=0,
        assetsOrder=None,
):
    """
    Mean-Variance portfolio for a target return
    Parameters
    ----------
    meanVec : Array
        A vector of mean returns of assets
    SigMat : Matrix
        A covariance matrix of appropriate dimensions
    retTarget : Float
        Target return percentage. Values specified between 0 and 100
    LongShort : Float
        Takes value between 0 and 1
    maxAlloc : Float
        Takes value between 0 and 1. Specifies the maximum weight an asset can get
    lambda_l1 : Float
        Takes a value greater than 0. Specifies L1 penalty
    lambda_l2 : Float
        Takes a value greater than 0. Specifies L2 penalty
    Returns
    -------
    w_opt : Array
        Returns the weights of given to each asset in form of a numpy array
    var_opt : Float
        Returns the variance of the portfolio
    """
    dailyRetTarget = retTarget
    minEret = min(meanVec)
    maxEret = max(meanVec)
    if (dailyRetTarget < minEret) or (maxEret < dailyRetTarget):
        part1 = minEret
        part2 = min(maxEret, dailyRetTarget)
        dailyRetTarget = max(part1, part2)

    d = sigMat.shape[0]
    if assetsOrder:
        temp = sigMat[:, assetsOrder]
        sigMat = temp[assetsOrder, :]
        meanVec = meanVec[assetsOrder]
    if lambda_l2:
        sigMat_l2 = sigMatShrinkage(sigMat, lambda_l2)
        sigMat_l2, e_min = SymPDcovmatrix(sigMat_l2)
    else:
        sigMat, e_min = SymPDcovmatrix(sigMat)
    # import pdb; pdb.set_trace()
    if longShort == 0:
        Aeq = np.ones(d)
        Beq = 1
        LB = np.zeros(d)
        UB = maxAlloc * np.ones(d)

        if assetsOrder:
            L_ine = np.hstack([-np.inf, -np.ones(d - 1)])
            tau = dailyRetTarget
            A = -meanVec
            B = -tau
            A = np.vstack([A, -1 * Dmat(d, 1)])
            B = np.hstack([B, np.zeros(d - 1)])
        else:
            tau = dailyRetTarget
            A = -meanVec
            B = -tau
            L_ine = -np.inf

        if lambda_l1:
            meanVec = -lambda_l1 * np.ones(d)
        else:
            meanVec = -np.zeros(d)

        if lambda_l2:
            P = sparse.csc_matrix(sigMat_l2)
        else:
            P = sparse.csc_matrix(sigMat)

        A = np.vstack([A, Aeq, np.eye(d)])
        l = np.hstack([L_ine, Beq, LB])
        u = np.hstack([B, Beq, UB])
        A = sparse.csc_matrix(A)

        prob = osqp.OSQP()
        # Setup workspace
        prob.setup(P, -meanVec, A, l, u, verbose=False, max_iter=10000, eps_abs=1e-8, eps_rel=1e-8, eps_prim_inf=1e-8,
                   eps_dual_inf=1e-8)
        # Solve problem
        res = prob.solve()
        w_opt = res.x
        if not w_opt.all():
            w_opt = np.ones(d) / d

    elif longShort != 0:
        A = np.hstack([np.zeros(d), np.ones(d), np.zeros(d)])
        B = 1 + abs(longShort)
        Grenze = min(abs(longShort), maxAlloc)

        if assetsOrder:
            tau = dailyRetTarget
            A = np.vstack([A, np.hstack([-meanVec, np.zeros(2 * d)])])
            B = np.hstack([B, -tau])
            A = np.vstack([A, np.hstack([-1 * Dmat(d, 1), np.zeros((d - 1, 2 * d))])])
            B = np.hstack([B, np.zeros(d - 1)])
            L_ine = np.hstack([0, -np.inf, -(1 + 2 * Grenze) * np.ones(d - 1)])
        else:
            tau = dailyRetTarget
            A = np.vstack([A, np.hstack([-meanVec, np.zeros(2 * d)])])
            B = np.hstack([B, -tau])
            L_ine = np.hstack([0, -np.inf])

        Aeq = np.vstack(
            [
                np.hstack([np.eye(d), -np.eye(d), np.eye(d)]),
                np.hstack([np.ones((1, d)), np.zeros((1, d)), np.zeros((1, d))])
            ]
        )
        Beq = np.hstack([np.zeros(d), 1])
        LB = np.hstack([-Grenze * np.ones(d), np.zeros(2 * d)])
        UB = maxAlloc * np.ones(3 * d)

        if lambda_l2:
            sigMat3d = np.vstack(
                [np.hstack([sigMat_l2, np.zeros((d, 2 * d))]), np.zeros((2 * d, 3 * d))]
            )
        else:
            sigMat3d = np.vstack(
                [np.hstack([sigMat, np.zeros((d, 2 * d))]), np.zeros((2 * d, 3 * d))]
            )

        sigMat3d = sigMat3d + np.diag(
            np.hstack([-0.1 * e_min * np.ones(d), 0.1 * e_min * np.ones(2 * d)])
        )

        if lambda_l1:
            meanvec3d = np.hstack([np.zeros(d), -lambda_l1 * np.ones(2 * d)])
        else:
            meanvec3d = np.hstack([np.zeros(d), np.zeros(2 * d)])

        A = np.vstack([A, Aeq, np.eye(3 * d)])
        l = np.hstack([L_ine, Beq, LB])
        u = np.hstack([B, Beq, UB])
        A = sparse.csc_matrix(A)
        sigMat3d = sparse.csc_matrix(sigMat3d)
        prob = osqp.OSQP()
        # Setup workspace
        prob.setup(sigMat3d, -meanvec3d, A, l, u, verbose=False, eps_abs=1e-8, eps_rel=1e-8, eps_prim_inf=1e-8,
                   eps_dual_inf=1e-8)
        # Solve problem
        res = prob.solve()
        wuv_opt = res.x
        if not wuv_opt.all():
            w_opt = np.ones(d) / d
        else:
            w_opt = wuv_opt[:d]
    t = np.dot(w_opt, sigMat)
    Var_opt = np.dot(t, w_opt.transpose())
    if assetsOrder:
        w_opt = w_opt[assetsOrder]
    # if exitflag!=1:
    # print("minimumVariancePortfolio: Exitflag different than 1 in quadprog")
    return w_opt, Var_opt


def unconstrained_mean_variance(M, Sigma):
    """
    Pure linear algebra solution for a mean variance portfolio optimization problem without constraints

    Parameters
    ----------
    M: numpy.ndarray
        Vector of expected returns for assets
    Sigma: numpy.ndarray
        Covariance matrix for assets

    Returns
    -------
    efficient_sigma: function
        see documentation below
    efficient_portfolio: function
        see documentation below
    """
    logger.debug(Sigma.shape)
    logger.debug(M.shape)

    SigmaInv = np.linalg.inv(Sigma)
    n = M.shape[0]
    O = np.ones((n, 1))
    k = M.T @ SigmaInv @ O
    l = M.T @ SigmaInv @ M
    m = O.T @ SigmaInv @ O
    g = (SigmaInv @ O * l - SigmaInv @ M * k) / (l * m - k ** 2)
    h = (SigmaInv @ M * m - SigmaInv @ O * k) / (l * m - k ** 2)
    a = h.T @ Sigma @ h
    b = 2 * g.T @ Sigma @ h
    c = g.T @ Sigma @ g

    def efficient_sigma(mu):
        """
        Return standard deviation / volatility of the efficient portfolio corresponding to
        target return mu

        Parameters
        ----------
        mu: float
            target return

        Returns
        -------
        out: float
            efficient portfolio volatility
        """
        return np.sqrt(a * mu ** 2 + b * mu + c)

    def efficient_portfolio(target_return):
        """
        Return asset allocations/weights for the
        portfolio corresponding to target_return

        Parameters
        ----------
        target_return: float
            target return for the efficient portfolio

        Returns
        -------
        out: numpy.ndarray
            efficient portfolio allocations for each asset
        """
        return g * target_return + h

    return efficient_sigma, efficient_portfolio


def _create_portfolio_grid(
    numPortGrid,
    meanVec,
    sigMat,
    startPort,
    numPort,
    shrinkage=0,
    minMu=None,
    maxMu=None,
):
    """Creates possible portfolios that can be invested in as part of a strategy

    Parameters
    ----------

    numPortGrid: int
        Number of total portfolios to generate in the frontier
    meanVec: array_like
        Vector of expected returns
    sigMat: numpy.ndarray
        Covariance matrix
    startPort: int
        First portfolio to use
    numPort: int
        Number of portfolios to pick from the frontier
    minMu: float
        Minimum expected return for portfolios in frontier
    maxMu: float
        Maximum expected return for portfolios in frontier

    Returns
    -------
    portfolios: numpy.ndarray
        Array of numPortfolios x 2. First column is exp. ret. 2nd column is vol.
    weights: numpy.ndarray
        Array of numPortfolios x numAssets. Allocations for each asset on each portfolio.
    """
    if not minMu:
        minMu = max(min(meanVec), 0) * 252

    if not maxMu:
        maxMu = max(meanVec) * 252

    mu = np.linspace(minMu, maxMu, numPortGrid)

    if shrinkage:
        sigMat = sigMatShrinkage(sigMat, shrinkage)

    # Two lines below can be used for debugging to match Das & Varma's results exactly.
    # in practice this optimization is not used often

    # eff_sigma, eff_port = unconstrained_mean_variance(meanVec, sigMat)
    # sigma = eff_sigma(mu)

    weights = []
    sigma = []
    for mu_i in mu:
        w, var_i = meanVariancePortfolioReturnsTarget(
            meanVec.squeeze() * 100, sigMat * 100, mu_i * 100, 0, lambda_l1=0.5
        )
        weights.append(np.clip(w, 0, 1))
        sigma.append(np.sqrt(var_i * 252 / 100))

    sigma = np.array(sigma)
    weights = np.array(weights)
    portfolios = [p for p in zip(mu.squeeze(), sigma.squeeze())]
    portfolios = np.array(portfolios[startPort : startPort + numPort])

    return portfolios, weights


def dynamic_programming_portfolio(
    meanVec,
    sigMat,
    invHorizon=10,
    initialWealth=100,
    wealthGoal=200,
    cashInjection=0,
    timeStep=1,
    numPortfolios=15,
    gridGranularity=10,
    gridMaxRet=None,
    shrinkage=0,
):
    """
    Dynamic programming solution for goal based investment based on
    "Dynamic Goals-Based Wealth Management using Reinforcement Learning (Das, s; Varma, S ;2020)"

    Parameters
    ----------
    meanVec: numpy.ndarray
        vector of expected returns for each asset
    sigMat: numpy.ndarray
        covariance matrix for the asset returns
    invHorizon: int
        investment horizon (in years) for the RL problem
    initialWealth: float
        initial wealth for the RL problem
    wealthGoal: float
        target wealth for the reward function of the RL problem
    cashInjection: float
        periodic cash injections into the investment
    timeStep: float
        time step ((dt) in years) for each decision made before getting to the time horizon
    numPortfolios: int
        number of portfolios to choose from in the efficient frontier
    gridGranularity: int
        number of state values - 1 considered at each time step
    gridMaxRet: Float
        Optional. Maximum return to use for possible portfolios
    shrinkage: Float
        Optional. Shrinkage coefficient for covariance matrix

    Returns
    -------
    weight_function: Function
       see create_wealth_function for details
    V: numpy.ndarray
        Value tensor result of the DP algorithm
    """

    firstPortfolio = 1
    extraPortfolios = 5
    numPeriods = invHorizon * int(1 / timeStep)
    portfolios, weights = _create_portfolio_grid(
        numPortfolios + extraPortfolios,
        meanVec,
        sigMat,
        firstPortfolio,
        numPortfolios,
        shrinkage=shrinkage,
        maxMu=gridMaxRet,
    )
    logger.debug(portfolios)

    exp_V = np.vectorize(_expected_value, signature="(),(n),(n),(),(m)->()")
    max_vec = np.vectorize(lambda x: 1 if x >= wealthGoal else 0)
    W = _create_wealth_grid(
        initialWealth, cashInjection, invHorizon, gridGranularity, timeStep, portfolios
    )
    grdPoints = len(W)

    logger.debug(W)
    V = np.zeros((grdPoints, numPeriods + 1))
    P = np.zeros((grdPoints, numPeriods), dtype=int)
    V[:, numPeriods] = max_vec(W)

    for t in range(1, numPeriods):
        for i, s_t in enumerate(W):
            t_exp_value = exp_V(s_t, W, V[:, numPeriods - t + 1], timeStep, portfolios)
            V[i, numPeriods - t] = t_exp_value.max()
            P[i, numPeriods - t] = max(np.where(t_exp_value == V[i, numPeriods - t])[0])
    V[:, 0] = exp_V(initialWealth, W, V[:, 1], timeStep, portfolios).max()
    P[:, 0] = exp_V(initialWealth, W, V[:, 1], timeStep, portfolios).argmax()

    weight_function = _create_weight_function(W, P, weights)

    return weight_function, V


def q_learning_portfolio(
    meanVec,
    sigMat,
    invHorizon=10,
    initialWealth=100,
    wealthGoal=200,
    cashInjection=0,
    timeStep=1,
    numPortfolios=15,
    gridGranularity=10,
    gridMaxRet=None,
    shrinkage=0,
    hParams=None,
    Q=None,
    returns=None,
):
    """
    This function uses the reinforcement learning (RL) approach described in
    "Dynamic Goals-Based Wealth Management using Reinforcement Learning (Das, s; Varma, S ;2020)"
    to choose a portfolio in the efficient frontier for every time step

    Parameters
    ----------
    meanVec: numpy.ndarray
        vector of expected returns for each asset
    sigMat: numpy.ndarray
        covariance matrix for the asset returns
    invHorizon: int
        investment horizon (in years) for the RL problem
    initialWealth: float
        initial wealth for the RL problem
    wealthGoal: float
        target wealth for the reward function of the RL problem
    cashInjection: float
        periodic cash injections into the investment
    timeStep: float
        time step ((dt) in years) for each decision made before getting to the time horizon
    numPortfolios: int
        number of portfolios to choose from in the efficient frontier
    gridGranularity: int
        number of state values - 1 considered at each time step
    gridMaxRet: Float
        Optional. Maximum return to use for possible portfolios
    shrinkage: Float
        Optional. Shrinkage coefficient for covariance matrix
    Q: numpy.ndarray
        Optional. Initial values for value tensor
    returns: numpy.ndarray
        Optional. Array of num_periods x num_assets daily log-returns.
        If provided period returns are sampled from empirical distribution instead of
        GBM based on meanVec and sigMat
    hParams: dict
        hyper-parameters used to train the RL strategy. See Das & Varma for more details

    Returns
    -------
    weight_function: Function
       see create_wealth_function for details
    Q: numpy.array
        Value function trained by the RL algorithm
    """
    if hParams is None:
        hParams = {"epsilon": 0.3, "alpha": 0.1, "gamma": 1, "epochs": 100}

    # This portion of the code selects equally spaced portfolios in the efficient frontier
    # the "+ 5" is only to replicate the paper but no really necessary
    firstPortfolio = 1
    extraPortfolios = 5
    numPeriods = invHorizon * int(1 / timeStep)
    portfolios, weights = _create_portfolio_grid(
        numPortfolios + extraPortfolios,
        meanVec,
        sigMat,
        firstPortfolio,
        numPortfolios,
        shrinkage=shrinkage,
        maxMu=gridMaxRet,
    )
    logger.debug(portfolios)

    # Generating possible states following a geometric brownian motion
    W = _create_wealth_grid(
        initialWealth, cashInjection, invHorizon, gridGranularity, timeStep, portfolios
    )
    gridPoints = len(W)
    logger.debug(W)

    maxVec = np.vectorize(lambda x: 1 if x >= wealthGoal else 0)

    if Q is None:
        Q = np.zeros((gridPoints, numPeriods + 1, numPortfolios))
    R = np.zeros((gridPoints, numPeriods + 1, numPortfolios))
    R[:, numPeriods, :] = np.broadcast_to(
        np.array([maxVec(W)]).T, (gridPoints, numPortfolios)
    )

    for e in range(hParams["epochs"]):
        Q = update_policy_path(
            Q,
            R,
            numPeriods,
            timeStep,
            initialWealth,
            W,
            portfolios,
            hParams,
            returns,
            weights,
        )

    P = Q.argmax(axis=2)
    weight_function = _create_weight_function(W, P, weights)

    return weight_function, Q


def rollingwindow_backtest(
    optimizerName,
    data,
    window_size,
    rebalance_time,
    maxAlloc=1,
    riskAversion=0,
    meanQuantile=0,
    retTarget=0,
    longShort=0,
    lambda_l1=0,
    lambda_l2=0,
    assetsOrder=None,
    initialWealth=100,
    wealthGoal=200,
    cashInjection=0,
    invHorizon=10,
    stratUpdateFreq=12,
    numPortOpt=15,
    gridGranularity=10,
    useEmpDist=False,
    hParams = None
):
    """
    function do the rolling window back test

    Parameters
    ----------
    optimizerName : String
        The name of the optimizer to use for rolling window exercise
    data : Dictionary
        Data with Ticker, Date and Adjusted Close price
    window_size : int
        parameter for the size of rolling window (>=2)
    rebalance_time : int
        rebalance time of rolling window test
    maxAlloc : Float
        maximum allocation. Takes values between 0 and 1
    riskAversion : Float
        Risk Aversion for your portfolio. Takes values greater than 0
    meanQuantile : Float
        Takes values between 0 and 1
    RetTarget : Float
        Target returns in percentage for optimizer. Takes values between 0 and 100
    LongShort : Float
        Takes value between 0 and 1
    maxAlloc : Float
        Takes value between 0 and 1. Specifies the maximum weight an asset can get
    lambda_l1 : Float
        Takes a value greater than 0. Specifies L1 penalty
    lambda_l2 : Float
        Takes a value greater than 0. Specifies L2 penalty
    initialWealth: Float
        Starting wealth for the dynamic programming case
    wealthGoal: Float
        Final target wealth
    cashInjection: Float
        Periodic cash injections for the investment goal. Period corresponds to rebalance time.
    invHorizon: int
        Number of year until target
    stratUpdateFreq: int
        Number of rebalance periods before updating strategy
    numPortOpt: int
        Number of portfolio options for DP and RL
    gridGranularity: int
        Number of wealth points to have in the wealth grid for every year. See Das & Varma (2020) for details.
    useEmpDist: bool
        If True the q_learning algorithm samples from historical returns instead
        of generating a return from GBM

    Returns
    -------
    R : 2d array
        return matrix depends on the rebalance time
    logret: 2d array
        log return matrix for each stocks
    w_all: 2d array
        optimal weight for each rebalance time
    rownames: array
        date time of rolling window test

    Notes
    -------
    Note for now we have provided additional parameters that'll be used in future versions of the optimizers
    """
    assert (
        window_size >= 2
    ), "At least 2 observations are needed to compute a covariance matrix"

    if isinstance(data, dict):
        df = pd.DataFrame(data)
        df.columns = ["date", "ticker", "price"]
        df1 = df.pivot_table(index="date", columns="ticker", values="price")
    elif isinstance(data, pd.DataFrame):
        df1 = data
    else:
        ValueError("data type not supported")

    df_logret = np.log(df1).diff().dropna(how="all")
    logret = df_logret.values
    n = logret.shape[0]
    d = rebalance_time
    start = window_size
    R = None
    currentWealth = initialWealth
    portfolio_return = None
    w_all = None
    Q = None

    for rebalCount, i in enumerate(range(start, n, d)):
        logger.info(f"Rebalance number {rebalCount} on day {i}")
        k = 0
        w_opt = np.zeros(df1.shape[1])
        
        window = df_logret[i - window_size : i].copy().dropna(axis=1)
        sample_stocks = window.columns
        logret_window = window.values
        sigMat = np.cov(logret_window, rowvar=False)
        meanVec = np.mean(logret_window, axis=0)

        if optimizerName == "minimumVariancePortfolio":
            w_sample, _ = minimumVariancePortfolio(
                sigMat,
                float(maxAlloc),
                float(longShort),
                float(lambda_l1),
                float(lambda_l2),
            )

        elif optimizerName == "meanVariancePortfolioReturnsTarget":
            w_sample, _ = meanVariancePortfolioReturnsTarget(
                meanVec,
                sigMat,
                float(retTarget),
                float(maxAlloc),
                float(longShort),
                float(lambda_l1),
                float(lambda_l2),
            )
        elif optimizerName == "dynamic_programming":
            if rebalCount % stratUpdateFreq == 0:
                strat_sample_stocks = sample_stocks
                w_func, prob = dynamic_programming_portfolio(
                    meanVec,
                    sigMat,
                    invHorizon=max(
                        invHorizon - math.ceil(rebalCount * rebalance_time / 252), 1
                    ),
                    initialWealth=currentWealth,
                    wealthGoal=wealthGoal,
                    cashInjection=cashInjection,
                    timeStep=rebalance_time / 252,
                    numPortfolios=numPortOpt,
                    gridGranularity=gridGranularity,
                    shrinkage=lambda_l2,
                )
                logger.info(f"Probability of success {prob[0, 0]*100:.2f}%")
            sample_stocks = strat_sample_stocks
            w_sample = w_func(currentWealth, rebalCount % stratUpdateFreq)

        elif optimizerName == "q_learning":
            if rebalCount % stratUpdateFreq == 0:
                strat_sample_stocks = sample_stocks
                if useEmpDist:
                    returns = window
                else:
                    returns = None
                w_func, Q = q_learning_portfolio(
                    meanVec,
                    sigMat,
                    invHorizon=max(
                        invHorizon - math.ceil(rebalCount * rebalance_time / 252), 1
                    ),
                    initialWealth=currentWealth,
                    wealthGoal=wealthGoal,
                    cashInjection=cashInjection,
                    timeStep=rebalance_time / 252,
                    numPortfolios=numPortOpt,
                    gridGranularity=gridGranularity,
                    shrinkage=lambda_l2,
                    Q=Q,
                    returns=returns,
                    hParams=hParams
                )
                logger.info(f"Probability of success {Q[:, 0, :].max()*100:.2f}%")
            sample_stocks = strat_sample_stocks
            w_sample = w_func(currentWealth, rebalCount % stratUpdateFreq)

        elif optimizerName == "G-learning":
            if i == start:
                g_learner = g_learn(
                    num_steps=12, num_risky_assets=logret.shape[1],
                    x_vals_init=1000*np.ones(logret.shape[1]) / logret.shape[1]
                )
            w_sample, g_learner = g_learn_rolling(
                t=int((i-start)/d % g_learner.num_steps), g_learner=g_learner,
                exp_returns=meanVec*d, sigma=sigMat*d,
                returns=df_logret.iloc[i-d:i].sum(axis=0).values / 100
            )

        else:
            raise ValueError(f"Optimization type {optimizerName} not defined")

        for j in range(df1.shape[1]):
            if df1.columns[j] in sample_stocks:
                w_opt[j] = w_sample[k]
                k += 1

        if w_all is None:
            w_all = w_opt
        else:
            w_all = np.vstack([w_all, w_opt])

        if (i + d) < n:
            logret_sample = np.nan_to_num(logret[i : i + d], nan=0)
            simple_returns = np.exp(logret_sample) - 1
            if R is None:
                R = w_opt.dot(simple_returns.T)
            else:
                R = np.hstack([R, w_opt.dot(simple_returns.T)])

        elif (i + d) >= n:
            logret_sample = np.nan_to_num(logret[i:], nan=0)
            simple_returns = np.exp(logret_sample) - 1
            R = np.hstack([R, w_opt.dot(simple_returns.T)])

        currentWealth = initialWealth * (1 + R).cumprod()[-1]

    rownames = df1.index[start + 1 :]
    R *= 100
    df_logret *= 100
    return R, df_logret, w_all, rownames


if __name__ == "__main__":
    pass
