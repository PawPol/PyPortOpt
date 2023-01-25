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
from ._utils import _create_wealth_grid, _create_weight_function, _expected_value
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


def nearestPD(A):
    """
    Find the nearest positive-definite matrix to input
    """
    if isPD(A):
        return A
    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


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


def constrain_matrix(
    d,
    meanvariance=0,
    maxShar=0,
    meanVec=None,
    riskfree=0,
    assetsOrder=None,
    maxAlloc=1,
    longShort=0,
    lambda_l1=0,
    turnover=None,
    w_pre=None,
    individual=False,
    exposure_constrain=0,
    w_bench=None,
    factor_exposure_constrain=None,
    U_factor=None,
    general_linear_constrain=None,
    U_genlinear=0,
    w_general=None,
):

    """
    This function creates the constraint matrices for an optimization problem, given a set of parameters. 
    
    Parameters:
    ----------
    d: int
        Number of assets
    meanvariance: int or float, optional
        Mean-variance constraint for the portfolio, defaults to 0
    maxShar: int or float, optional
        Maximum sharpe ratio constraint for the portfolio, defaults to 0
    meanVec: array-like, optional
        Mean return vector of assets, defaults to None
    riskfree: int or float, optional
        Risk free rate, defaults to 0
    assetsOrder: array-like, optional
        assets ordering constraints, defaults to None
    maxAlloc: int or float, optional
        Maximum allocation in a single asset, defaults to 1
    longShort: int or float, optional
        Long-Short constraint for the portfolio, defaults to 0
    lambda_l1: int or float, optional
        L1 regularization constraint, defaults to 0
    turnover: array or float, optional
        Turnover constraint, defaults to None
    w_pre: array-like, optional
        Initial weights, defaults to None
    exposure_constrain: int or float, optional
        Exposure constraint, defaults to 0
    w_bench: array-like, optional
        Weights of benchmark portfolio, defaults to None
    factor_exposure_constrain: array-like, optional
        Factor exposure constraints, defaults to None
    U_factor: array or float, optional
        Upper bound on factor exposure, defaults to None
    general_linear_constrain: array-like, optional
        General linear constraint, defaults to None
    U_genlinear: int or float, optional
        Upper bound on general linear constraint, defaults to 0
    w_general: array-like, optional
        Weights of general linear constraint, defaults to None
    
    Returns:
    -------
    A, l, u: tuple of arrays
        Constraint matrices for the optimization problem
    """

    # factor_exposure_constrain 1xd vector for time t
    if longShort == 0:
        Aeq = np.ones(d)  # sum(w) = 1
        Beq = 1
        LB = np.zeros(d)
        UB = maxAlloc * np.ones(d)

        A = np.vstack([Aeq, np.eye(d)])  # 0 < w_i < maxAlloc
        l = np.hstack([Beq, LB])
        u = np.hstack([Beq, UB])
        if maxShar:
            A = np.vstack(
                [
                    A,  # sum(w) = kappa
                    -np.eye(d),  # w_i >= 0
                    np.zeros(d),  # kappa > 0
                    meanVec,
                ]
            )  # kappa*w'mu = 1
            Bwuv = np.hstack(
                [1, maxAlloc * UB, LB, 1, 0]
            )  # [w kappa] where kappa>0 is a scalar for the rescaling
            l = np.hstack([0, -np.inf * np.ones(2 * d + 1), 1])
            u = np.hstack([np.zeros(2 * d + 1), -1e-12, 1])

        if assetsOrder:
            # ordering constraint w_1 >= w_2 >= w_3 >= ... >= w_d
            L_ine = np.hstack([-np.inf, -np.ones(d - 1)])
            A = np.vstack([A, -1 * Dmat(d, 1)])
            B = np.zeros(d - 1)
            if maxShar:
                Bwuv = np.hstack([Bwuv, np.zeros(d - 1)])

            l = np.hstack([l, -np.ones(d - 1)])
            u = np.hstack([u, B])

        if meanvariance and meanVec.any():
            if riskfree:
                A = np.vstack([A, -meanVec + riskfree])  # w'mu + (1-w)rf > meanvariance
                l = np.hstack([np.zeros(d + 1), -np.inf])
                u = np.hstack([u, -meanvariance + riskfree])
            else:
                A = np.vstack([A, -meanVec])  # w' * mu > meanvariance
                l = np.hstack([l, -np.inf])
                u = np.hstack([u, -meanvariance])

        if U_factor is not None and factor_exposure_constrain is not None:
            # factor_exposure_constrain can be a d x k matrix and U_factor can be a k length vector
            A = np.vstack([A, factor_exposure_constrain])  # abs(beta_k' * w) < U
            l = np.hstack([l, -U_factor])
            u = np.hstack([u, U_factor])
            if maxShar:
                Bwuv = np.hstack([Bwuv, 0])

        if U_genlinear > 0 and general_linear_constrain is not None:
            # A_B (w − w_B ) ≤ u_B
            A = np.vstack([A, general_linear_constrain])
            l = np.hstack([l, 0])
            u = np.hstack(
                [
                    u,
                    U_genlinear
                    + sum(np.dot(general_linear_constrain, w_general.reshape(-1, 1))),
                ]
            )
            if maxShar:
                Bwuv = np.hstack([Bwuv, 0])

        if turnover is not None and w_pre is not None:
            # expend to [w, w_p, w_n]
            # w_p: w_old-w(w-w_old > 0)
            # w_n: w-w_old(w-wold < 0)
            if individual == True:
                # turnover is a vector abs(w_old_i - w_i) < U_i
                A = np.hstack([A, np.zeros((len(A), 2 * d))])
                A = np.vstack(
                    [
                        A,
                        np.hstack(
                            [np.eye(d), np.eye(d), -np.eye(d)]
                        ),  # w + w_p - w_n = w_old
                        np.hstack(
                            [np.zeros((d, d)), np.eye(d), np.eye(d)]
                        ),  # abs(w_old-w) < turnover
                        np.hstack(
                            [np.zeros((d, d)), np.eye(d), np.zeros((d, d))]
                        ),  # w_p_i < turnover
                        np.hstack(
                            [np.zeros((d, d)), np.zeros((d, d)), np.eye(d)]
                        ),  # w_n_i < turnover
                    ]
                )
                l = np.hstack([l, w_pre, np.zeros(d), np.zeros(2 * d)])
                u = np.hstack([u, w_pre, turnover, turnover * np.ones(2 * d)])
                if maxShar:
                    Bwuv = np.hstack([Bwuv, np.zeros(4 * d)])
            else:
                # turnover is a float sum(w_old - w) < U
                A = np.hstack([A, np.zeros((len(A), 2 * d))])
                A = np.vstack(
                    [
                        A,
                        np.hstack([np.eye(d), np.eye(d), -np.eye(d)]),
                        np.hstack([np.zeros(d), np.ones(d), np.ones(d)]),
                        np.hstack([np.zeros((d, d)), np.eye(d), np.zeros((d, d))]),
                        np.hstack([np.zeros((d, d)), np.zeros((d, d)), np.eye(d)]),
                    ]
                )
                l = np.hstack([l, w_pre, 0, np.zeros(2 * d)])
                u = np.hstack([u, w_pre, turnover, turnover * np.ones(2 * d)])
                if maxShar:
                    Bwuv = np.hstack([Bwuv, np.zeros(3 * d + 1)])

        if exposure_constrain > 0 and w_bench is not None:
            # sum(abs(w - w_bench)) < U
            A = np.hstack([A, np.zeros((len(A), 2 * d))])
            A = np.vstack(
                [
                    A,
                    np.hstack([np.eye(d), np.eye(d), -np.eye(d)]),
                    np.hstack([np.zeros(d), np.ones(d), np.ones(d)]),
                    np.hstack([np.zeros((d, d)), np.eye(d), np.zeros((d, d))]),
                    np.hstack([np.zeros((d, d)), np.zeros((d, d)), np.eye(d)]),
                ]
            )
            l = np.hstack([l, w_bench, 0, np.zeros(2 * d)])
            u = np.hstack(
                [u, w_bench, exposure_constrain, exposure_constrain * np.ones(2 * d)]
            )
            if maxShar:
                Bwuv = np.hstack([Bwuv, np.zeros(3 * d + 1)])

    else:
        # the following two auxiliary variables are introduced for the long-short portfolio estimation
        # u = w.*(w>0) % postive part of w
        # v = -1*(w.*(w<0)) % negative part of w
        A = np.hstack([np.zeros(d), np.ones(d), np.zeros(d)])
        B = 1 + abs(longShort)  # sum of u's <= 1+longShort
        Grenze = min(abs(longShort), maxAlloc)
        Aeq = np.vstack(
            [
                np.hstack([np.eye(d), -np.eye(d), np.eye(d)]),  # w - u + v = 0
                np.hstack([np.ones(d), np.zeros(d), np.zeros(d)]),  # sum(w) = 1
            ]
        )
        Beq = np.hstack([np.zeros(d), 1])
        LB = np.hstack([-Grenze * np.ones(d), np.zeros(2 * d)])  # w >= -Grenze
        UB = (1 + Grenze) * np.ones(3 * d)  # [w,u,v] <= (1+Grenze)
        A = np.vstack([Aeq, A, np.eye(3 * d)])
        l = np.hstack([Beq, 0, LB])
        u = np.hstack([Beq, B, UB])
        if maxShar:
            A = np.vstack(
                [
                    A,
                    -np.eye(3 * d),
                    np.zeros(3 * d),
                    np.hstack([meanVec, np.zeros(2 * d)]),
                ]
            )
            Bwuv = np.hstack(
                [np.zeros(d), 1, (1 + Grenze), UB, -LB, 1, 0]
            )  # [w u v kappa] where kappa>0 is a scalar for the rescaling
            l = np.hstack([np.zeros(d + 1), -np.inf * np.ones(6 * d + 2), 1])
            u = np.hstack([np.zeros(7 * d + 2), -1e-12, 1])

        if assetsOrder:
            A = np.vstack([A, np.hstack([-1 * Dmat(d, 1), np.zeros((d - 1, 2 * d))])])
            if maxShar:
                Bwuv = np.hstack([Bwuv, np.zeros(d - 1)])
            l = np.hstack([l, -(1 + 2 * Grenze) * np.ones(d - 1)])
            u = np.hstack([u, np.zeros(d - 1)])

        if meanvariance and meanVec.any():
            if riskfree:
                A = np.vstack(
                    [A, np.hstack([-meanVec + riskfree, np.zeros(2 * d)])]
                )  # w'mu + (1-w)rf > meanvariance
                l = np.hstack([np.zeros(d + 1), 0, LB, -np.inf])
                u = np.hstack([u, -meanvariance + riskfree])
            else:
                A = np.vstack(
                    [A, np.hstack([-meanVec, np.zeros(2 * d)])]
                )  # w' * mu > meanvariance
                l = np.hstack([l, -np.inf])
                u = np.hstack([u, -meanvariance])

        if U_factor is not None and factor_exposure_constrain is not None:
            # factor_exposure_constrain can be a d x k matrix and U_factor can be a k length vector
            A = np.vstack(
                [A, np.hstack([factor_exposure_constrain, np.zeros(2 * d)])]
            )  # abs(beta_k' * w) < U
            l = np.hstack([l, -U_factor])
            u = np.hstack([u, U_factor])
            if maxShar:
                Bwuv = np.hstack([Bwuv, 0])
        if U_genlinear > 0 and general_linear_constrain is not None:
            # A_B (w − w_B ) ≤ u_B
            A = np.vstack([A, np.hstack([general_linear_constrain, np.zeros(2 * d)])])
            l = np.hstack([l, 0])
            u = np.hstack(
                [
                    u,
                    U_genlinear
                    + sum(np.dot(general_linear_constrain, w_general.reshape(-1, 1))),
                ]
            )
            if maxShar:
                Bwuv = np.hstack([Bwuv, 0])

        if turnover is not None and w_pre is not None:
            # expend to [w, w_p, w_n]
            # w_p: w_old-w(w-w_old > 0)
            # w_n: w-w_old(w-wold < 0)
            if individual == True:
                A = np.hstack([A, np.zeros((len(A), 2 * d))])
                A = np.vstack(
                    [
                        A,
                        np.hstack(
                            [
                                np.eye(d),
                                np.zeros((d, d)),
                                np.zeros((d, d)),
                                np.eye(d),
                                -np.eye(d),
                            ]
                        ),
                        np.hstack(
                            [
                                np.zeros((d, d)),
                                np.zeros((d, d)),
                                np.zeros((d, d)),
                                np.eye(d),
                                np.eye(d),
                            ]
                        ),
                        np.hstack(
                            [
                                np.zeros((d, d)),
                                np.zeros((d, d)),
                                np.zeros((d, d)),
                                np.eye(d),
                                np.zeros((d, d)),
                            ]
                        ),
                        np.hstack(
                            [
                                np.zeros((d, d)),
                                np.zeros((d, d)),
                                np.zeros((d, d)),
                                np.zeros((d, d)),
                                np.eye(d),
                            ]
                        ),
                    ]
                )
                l = np.hstack([l, w_pre, np.zeros(3 * d)])
                u = np.hstack(
                    [
                        u,
                        w_pre,
                        turnover,
                        np.hstack([turnover, turnover]) * np.ones(2 * d),
                    ]
                )
                if maxShar:
                    Bwuv = np.hstack([Bwuv, np.zeros(4 * d)])
            else:
                A = np.hstack([A, np.zeros((len(A), 2 * d))])
                A = np.vstack(
                    [
                        A,
                        np.hstack(
                            [
                                np.eye(d),
                                np.zeros((d, d)),
                                np.zeros((d, d)),
                                np.eye(d),
                                -np.eye(d),
                            ]
                        ),
                        np.hstack(
                            [
                                np.zeros(d),
                                np.zeros(d),
                                np.zeros(d),
                                np.ones(d),
                                np.ones(d),
                            ]
                        ),
                        np.hstack(
                            [
                                np.zeros((d, d)),
                                np.zeros((d, d)),
                                np.zeros((d, d)),
                                np.eye(d),
                                np.zeros((d, d)),
                            ]
                        ),
                        np.hstack(
                            [
                                np.zeros((d, d)),
                                np.zeros((d, d)),
                                np.zeros((d, d)),
                                np.zeros((d, d)),
                                np.eye(d),
                            ]
                        ),
                    ]
                )
                l = np.hstack([l, w_pre, 0, np.zeros(2 * d)])
                u = np.hstack([u, w_pre, turnover, turnover * np.ones(2 * d)])
                if maxShar:
                    Bwuv = np.hstack([Bwuv, np.zeros(3 * d + 1)])

        if exposure_constrain > 0 and w_bench is not None:
            # sum(abs(w - w_bench)) < U
            A = np.hstack([A, np.zeros((len(A), 2 * d))])
            A = np.vstack(
                [
                    A,
                    np.hstack(
                        [
                            np.eye(d),
                            np.zeros((d, d)),
                            np.zeros((d, d)),
                            np.eye(d),
                            -np.eye(d),
                        ]
                    ),
                    np.hstack(
                        [np.zeros(d), np.zeros(d), np.zeros(d), np.ones(d), np.ones(d)]
                    ),
                    np.hstack(
                        [
                            np.zeros((d, d)),
                            np.zeros((d, d)),
                            np.zeros((d, d)),
                            np.eye(d),
                            np.zeros((d, d)),
                        ]
                    ),
                    np.hstack(
                        [
                            np.zeros((d, d)),
                            np.zeros((d, d)),
                            np.zeros((d, d)),
                            np.zeros((d, d)),
                            np.eye(d),
                        ]
                    ),
                ]
            )
            l = np.hstack([l, w_bench, 0, np.zeros(2 * d)])
            u = np.hstack(
                [u, w_bench, exposure_constrain, exposure_constrain * np.ones(2 * d)]
            )
            if maxShar:
                Bwuv = np.hstack([Bwuv, np.zeros(3 * d + 1)])

    if maxShar:
        # add kappa to constrain matrix
        A = np.hstack([A, -Bwuv.reshape(-1, 1)])

    return A, l, u


def penalty_vector(
    d,
    sigMat,
    maxShar=0,
    longShort=0,
    lambda_l1=0,
    turnover=None,
    exposure_constrain=0,
    TE_constrain=None,
    Q_b=None,
    Q_bench=None,
):
    """
    This function calculates the penalty vector for an optimization problem, given a set of parameters. 
    
    Parameters:
    ----------
    d: int
        Number of assets
    sigMat: array-like
        Covariance matrix of assets
    longShort: int or float, optional
        Long-Short constraint for the portfolio, defaults to 0
    lambda_l1: int or float, optional
        L1 regularization term, defaults to 0
    turnover: array or float, optional
        Turnover constraint, defaults to None
    exposure_constrain: int or float, optional
        Exposure constraint, defaults to 0
    TE_constrain: array-like, optional
        Tracking error constraint, defaults to None
    Q_b: array-like, optional
        Quadratic term for bias, defaults to None
    Q_bench: array-like, optional
        Benchmark for quadratic term, defaults to None
    
    Returns:
    -------
    meanVec: array
        Penalty vector
    """
    if longShort == 0:
        if lambda_l1:
            # lambda_l1 * w
            meanVec = -lambda_l1 * np.ones(d)
        else:
            meanVec = -np.zeros(d)
        if TE_constrain:
            # (w − wB )′Σ(w − wB )
            meanVec = meanVec + 2 * np.dot(TE_constrain, sigMat)
        if turnover is not None or exposure_constrain:
            # expend the weight vector
            meanVec = np.hstack([meanVec, np.zeros(2 * d)])
        if Q_b:
            meanVec = meanVec + 2 * np.dot(Q_bench, Q_b)
    else:
        if lambda_l1:
            # lambda_l1 * (u+v)
            meanVec = np.hstack([np.zeros(d), -lambda_l1 * np.ones(2 * d)])
        else:
            meanVec = np.hstack([np.zeros(d), np.zeros(2 * d)])
        if TE_constrain:
            meanVec = meanVec + np.hstack(
                [
                    np.zeros(d),
                    np.dot(TE_constrain, sigMat),
                    -np.dot(TE_constrain, sigMat),
                ]
            )
        if turnover is not None or exposure_constrain:
            meanVec = np.hstack([meanVec, np.zeros(2 * d)])
        if Q_b:
            meanVec = meanVec + np.hstack(
                [np.zeros(d), np.dot(Q_bench, Q_b), -np.dot(Q_bench, Q_b)]
            )
    if maxShar:
        meanVec = np.hstack([meanVec, 0])
    return meanVec


def sigMat_expend(
    d,
    sigMat,
    maxShar=0,
    longShort=0,
    turnover=None,
    exposure_constrain=0,
    TE_constrain=0,
    general_quad=0,
    Q_w=None,
    Q_b=None,
):
    """
    This function expands the covariance matrix for an optimization problem, given a set of parameters. 
    
    Parameters:
    ----------
    d: int
        Number of assets
    sigMat: array-like
        Covariance matrix of assets
    longShort: int or float, optional
        Long-Short constraint for the portfolio, defaults to 0
    turnover: array or float, optional
        Turnover constraint, defaults to None
    exposure_constrain: int or float, optional
        Exposure constraint, defaults to 0
    TE_constrain: int or float, optional
        Tracking error constraint, defaults to 0
    general_quad: int or float, optional
        General quadratic constraint, defaults to 0
    Q_w: array-like, optional
        Quadratic term for weights, defaults to None
    Q_b: array-like, optional
        Quadratic term for bias, defaults to None
    
    Returns:
    -------
    sigMat: array
        Expanded covariance matrix
    """
    if Q_w:
        sigMat += Q_w
    if Q_b:
        sigMat += Q_b
    if longShort == 0:
        if turnover is not None or exposure_constrain:
            sigMat = np.vstack(
                [np.hstack([sigMat, np.zeros((d, 2 * d))]), np.zeros((2 * d, 3 * d))]
            )
        if TE_constrain or general_quad:
            sigMat *= 2
    else:
        sigMat = np.vstack(
            [np.hstack([sigMat, np.zeros((d, 2 * d))]), np.zeros((2 * d, 3 * d))]
        )
        if turnover is not None or exposure_constrain:
            sigMat = np.vstack(
                [
                    np.hstack([sigMat, np.zeros((3 * d, 2 * d))]),
                    np.zeros((2 * d, 5 * d)),
                ]
            )
        if TE_constrain or general_quad:
            sigMat *= 2
    if maxShar:
        # add kappa
        sigMat = np.vstack(
            [
                np.hstack([sigMat, np.zeros((sigMat.shape[0], 1))]),
                np.zeros(sigMat.shape[0] + 1),
            ]
        )
    return sigMat


def portfolio_optimization(
    meanVec,
    sigMat,
    retTarget,
    longShort,
    maxAlloc=1,
    lambda_l1=0,
    lambda_l2=0,
    riskfree=0,
    assetsOrder=None,
    maxShar=0,
    turnover=None,
    w_pre=None,
    individual=False,
    exposure_constrain=0,
    w_bench=None,
    factor_exposure_constrain=None,
    U_factor=None,
    general_linear_constrain=None,
    U_genlinear=0,
    w_general=None,
    TE_constrain=0,
    general_quad=0,
    Q_w=None,
    Q_b=None,
    Q_bench=None,
):
    """
    function do the portfolio optimization

    Parameters
    ----------
    retTarget : Float
        Target returns in percentage for optimizer. Takes values between 0 and 100
    LongShort : Float
        Takes value between 0 and 1
    sigMat: array-like
        Covariance matrix of assets
    maxAlloc : Float
        Takes value between 0 and 1. Specifies the maximum weight an asset can get
    lambda_l1 : Float
        Takes a value greater than 0. Specifies L1 penalty
    lambda_l2 : Float
        Takes a value greater than 0. Specifies L2 penalty
    maxShar: int or float, optional
        Maximum sharpe ratio constraint for the portfolio, defaults to 0
    meanVec: array-like, optional
        Mean return vector of assets, defaults to None
    riskfree: int or float, optional
        Risk free rate, defaults to 0
    assetsOrder: array-like, optional
        assets ordering constraints, defaults to None
    individual : bool
        Individual turnover constrain, defaults to None
    turnover: array or float, optional
        Turnover constraint, defaults to None
    w_pre: array-like, optional
        Initial weights, defaults to None
    exposure_constrain: int or float, optional
        Exposure constraint, defaults to 0
    w_bench: array-like, optional
        Weights of benchmark portfolio, defaults to None
    factor_exposure_constrain: array-like, optional
        Factor exposure constraints, defaults to None
    U_factor: array or float, optional
        Upper bound on factor exposure, defaults to None
    general_linear_constrain: array-like, optional
        General linear constraint, defaults to None
    U_genlinear: int or float, optional
        Upper bound on general linear constraint, defaults to 0
    w_general: array-like, optional
        Weights of general linear constraint, defaults to None
    TE_constrain: int or float, optional
        Tracking error constraint, defaults to 0
    general_quad: int or float, optional
        General quadratic constraint, defaults to 0
    Q_w: array-like, optional
        Quadratic term for weights, defaults to None
    Q_b: array-like, optional
        Quadratic term for bias, defaults to None
    Q_bench: array-like, optional
        Benchmark for quadratic term, defaults to None
    Returns
    -------
    w_opt : Array
        Returns the weights of given to each asset in form of a numpy array
    var_opt : Float
        Returns the variance of the portfolio
    """
    if retTarget:
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
        if retTarget or maxShar:
            meanVec = meanVec[assetsOrder]
    if lambda_l2:
        sigMat_shrik = sigMatShrinkage(sigMat, lambda_l2)
        sigMat_shrik = nearestPD(sigMat_shrik)
    else:
        sigMat_shrik = nearestPD(sigMat)
    # import pdb; pdb.set_trace()

    A, l, u = constrain_matrix(
        d,
        retTarget,
        maxShar,
        meanVec,
        riskfree,
        assetsOrder,
        maxAlloc,
        longShort,
        lambda_l1,
        turnover,
        w_pre,
        individual,
        exposure_constrain,
        w_bench,
        factor_exposure_constrain,
        U_factor,
        general_linear_constrain,
        U_genlinear,
        w_general,
    )
    sigMat_exp = sigMat_expend(
        d,
        sigMat_shrik,
        maxShar,
        longShort,
        turnover,
        exposure_constrain,
        TE_constrain,
        general_quad,
        Q_w,
        Q_b,
    )
    meanVec = penalty_vector(
        d,
        sigMat_shrik,
        maxShar,
        longShort,
        lambda_l1,
        turnover,
        exposure_constrain,
        TE_constrain,
        Q_b,
        Q_bench,
    )

    P = sparse.csc_matrix(sigMat_exp)
    A = sparse.csc_matrix(A)
    prob = osqp.OSQP()
    # Setup workspace
    prob.setup(
        P,
        -meanVec,
        A,
        l,
        u,
        verbose=False,
        max_iter=10000,
        eps_abs=1e-8,
        eps_rel=1e-8,
        eps_prim_inf=1e-8,
        eps_dual_inf=1e-8,
    )
    # Solve problem
    res = prob.solve()
    w_opt = res.x
    if not w_opt.all():
        w_opt = np.ones(d) / d

    if maxShar:
        w_opt = w_opt[:d] / w_opt[-1]
    else:
        w_opt = w_opt[:d]
    Var_opt = np.dot(np.dot(w_opt, sigMat), w_opt.transpose())
    if assetsOrder:
        w_opt = w_opt[assetsOrder]

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
    meanQuantile=0,
    retTarget=0,
    longShort=0,
    lambda_l1=0,
    lambda_l2=0,
    initialWealth=100,
    wealthGoal=200,
    cashInjection=0,
    invHorizon=10,
    stratUpdateFreq=12,
    numPortOpt=15,
    gridGranularity=10,
    useEmpDist=False,
    hParams=None,
    g_steps=12,
    g_goal_rate=0.8,
    g_lambda=0.001,
    g_omega=1.0,
    g_eta=1.5,
    g_rho=0.4,
    g_beta=1000.0,
    g_gamma=0.95,
    riskfree=0,
    assetsOrder=None,
    maxShar=0,
    turnover=None,
    w_pre=None,
    individual=False,
    exposure_constrain=0,
    w_bench=None,
    factor_exposure_constrain=None,
    U_factor=None,
    general_linear_constrain=None,
    U_genlinear=0,
    w_general=None,
    TE_constrain=0,
    general_quad=0,
    Q_w=None,
    Q_b=None,
    Q_bench=None,
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
    initialWealth : Float
        Starting wealth for the dynamic programming case
    wealthGoal : Float
        Final target wealth
    cashInjection : Float
        Periodic cash injections for the investment goal. Period corresponds to rebalance time.
    invHorizon : int
        Number of year until target
    stratUpdateFreq : int
        Number of rebalance periods before updating strategy
    numPortOpt : int
        Number of portfolio options for DP and RL
    gridGranularity : int
        Number of wealth points to have in the wealth grid for every year. See Das & Varma (2020) for details.
    useEmpDist : bool
        If True the q_learning algorithm samples from historical returns instead
        of generating a return from GBM
    g_steps : int
        The RL model steps (in month)
    g_goal_rate : Float
        Benchmark portfolio growth rate
    g_lambda : Float
        The penalty strength of not reaching target portfolio value
    g_omega : Float
        The parameter approximating transaction costs
    g_eta : Float
        The parameter that defines the desired growth rate of the current portfolio
    g_rho : Float
        A relative weight of the portfolio-independent and portfolio-dependent terms
    g_beta : Float
        The parameter to determine the strength of entropy regularization
    g_gamma : Float
        The discount factor in accumulative reward function
    maxShar: int or float, optional
        Maximum sharpe ratio constraint for the portfolio, defaults to 0
    meanVec: array-like, optional
        Mean return vector of assets, defaults to None
    riskfree: int or float, optional
        Risk free rate, defaults to 0
    assetsOrder: array-like, optional
        assets ordering constraints, defaults to None
    individual : bool
        Individual turnover constrain, defaults to None
    turnover: array or float, optional
        Turnover constraint, defaults to None
    w_pre: array-like, optional
        Initial weights, defaults to None
    exposure_constrain: int or float, optional
        Exposure constraint, defaults to 0
    w_bench: array-like, optional
        Weights of benchmark portfolio, defaults to None
    factor_exposure_constrain: array-like, optional
        Factor exposure constraints, defaults to None
    U_factor: array or float, optional
        Upper bound on factor exposure, defaults to None
    general_linear_constrain: array-like, optional
        General linear constraint, defaults to None
    U_genlinear: int or float, optional
        Upper bound on general linear constraint, defaults to 0
    w_general: array-like, optional
        Weights of general linear constraint, defaults to None
    TE_constrain: int or float, optional
        Tracking error constraint, defaults to 0
    general_quad: int or float, optional
        General quadratic constraint, defaults to 0
    Q_w: array-like, optional
        Quadratic term for weights, defaults to None
    Q_b: array-like, optional
        Quadratic term for bias, defaults to None
    Q_bench: array-like, optional
        Benchmark for quadratic term, defaults to None
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
    ret = df1.values[1:] / df1.values[:-1] - 1
    ret = ret[~np.isnan(ret).all(axis=1)]
    n = logret.shape[0]
    d = rebalance_time
    start = window_size
    R = None
    currentWealth = initialWealth
    portfolio_return = None
    w_all = None
    Q = None

    for rebalCount, i in enumerate(range(start, n, d)):
        # logger.info(f"Rebalance number {rebalCount} on day {i}")
        k = 0
        w_opt = np.zeros(df1.shape[1])

        window = df_logret[i - window_size : i].copy().dropna(axis=1)
        sample_stocks = window.columns
        logret_window = window.values
        sigMat = np.cov(logret_window, rowvar=False)
        meanVec = np.mean(logret_window, axis=0)

        if (
            optimizerName == "minimumVariancePortfolio"
            or optimizerName == "meanVariancePortfolioReturnsTarget"
        ):
            w_sample, _ = portfolio_optimization(
                meanVec,
                sigMat,
                float(retTarget),
                float(longShort),
                float(maxAlloc),
                float(lambda_l1),
                float(lambda_l2),
                float(riskfree),
                assetsOrder,
                maxShar,
                turnover,
                w_pre,
                individual,
                float(exposure_constrain),
                w_bench,
                factor_exposure_constrain,
                U_factor,
                general_linear_constrain,
                float(U_genlinear),
                w_general,
                float(TE_constrain),
                float(general_quad),
                Q_w,
                Q_b,
                Q_bench,
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
                    hParams=hParams,
                )
                # logger.info(f"Probability of success {Q[:, 0, :].max()*100:.2f}%")
            sample_stocks = strat_sample_stocks
            w_sample = w_func(currentWealth, rebalCount % stratUpdateFreq)

        elif optimizerName == "G-learning":
            if i == start:
                g_learner = g_learn(
                    num_steps=g_steps,
                    rebalance_time=d,
                    num_risky_assets=logret.shape[1],
                    x_vals_init=initialWealth
                    * np.ones(logret.shape[1])
                    / logret.shape[1],
                    lambd=g_lambda,
                    omega=g_omega,
                    eta=g_eta,
                    rho=g_rho,
                    beta=g_beta,
                    gamma=g_gamma,
                    target_return=g_goal_rate,
                )
            if i + d <= n:
                w_sample, g_learner = g_learn_rolling(
                    t=int((i - start) / d % g_learner.num_steps),
                    g_learner=g_learner,
                    exp_returns=meanVec * d,
                    sigma=sigMat * d,
                    returns=np.cumprod(ret[i : i + d] + 1, axis=0) - 1,
                )
            else:
                d_final = n - i
                w_sample, g_learner = g_learn_rolling(
                    t=int((i - start) / d % g_learner.num_steps),
                    g_learner=g_learner,
                    exp_returns=meanVec * d_final,
                    sigma=sigMat * d_final,
                    returns=np.cumprod(ret[i : i + d_final] + 1, axis=0) - 1,
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
        w_pre = w_opt.reshape(1, -1)
        currentWealth = initialWealth * (1 + R).cumprod()[-1]

    rownames = df1.index[start + 1 :]
    R *= 100
    df_logret *= 100
    return R, df_logret, w_all, rownames


if __name__ == "__main__":
    pass
