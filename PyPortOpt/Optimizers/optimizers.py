"""
The PortOpt application is powered by multiple optimizers designed to implement theory in an elegant 
and easy to use way.

This module consists all the functions required to run a portfolio optimization using parameters 
that the user inputs
"""
import math
import numpy as np
from numpy import linalg as LA
import pandas as pd
import osqp
import scipy as sp
from scipy import sparse
from scipy.stats import norm


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
    data = pd.DataFrame.from_dict(data)
    df = data[["Date", "Ticker", "Adjusted_Close"]]
    df.columns = ["date", "ticker", "price"]
    df1 = df.pivot_table(index=["date"], columns="ticker", values=["price"])
    df1.columns = [col[1] for col in df1.columns.values]
    df_logret = 100 * (np.log(df1) - np.log(df1.shift(1)))
    df_logret = df_logret[1:]
    logret = np.array(df_logret)

    df_daily_returns = df1.pct_change()

    df_daily_returns = df_daily_returns[1:]
    data = np.array(data)
    daily_returns = np.array(df_daily_returns)
    n = logret.shape[0]
    sigMat = np.cov(logret, rowvar=False)
    meanVec = np.mean(logret, axis=0)
    return meanVec, sigMat


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
    t = np.dot(np.diag(sig ** (-1)), sigMat)
    corrMat = np.dot(t, np.diag(sig ** (-1)))
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
            np.eye(d) + (np.ones(d, d) - np.eye(d)) * np.mean(corrs),
        )
        sigMat = sigMat + lambda_l2 * np.dot(t, np.mean(sig) * np.eye(d))
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
        sigMat = sigMatShrinkage(sigMat, lambda_l2)
        sigMat, e_min = SymPDcovmatrix(sigMat)
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

        P = sparse.csc_matrix(sigMat)
        A = sparse.csc_matrix(A)

        prob = osqp.OSQP()
        # Setup workspace
        prob.setup(P, -meanVec, A, l, u, verbose=False)
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
        prob.setup(sigMat3d, -meanvec3d, A, l, u, verbose=False)
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
    dailyRetTarget = 100 * ((retTarget / 100 + 1) ** (1 / 250) - 1)
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
        sigMat = sigMatShrinkage(sigMat, lambda_l2)
        sigMat, e_min = SymPDcovmatrix(sigMat)
    else:
        sigMat, e_min = SymPDcovmatrix(sigMat)

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
            meanVec = -lambda_l1 * meanVec
        else:
            meanVec = -np.zeros(d)

        A = np.vstack([A, Aeq, np.eye(d)])
        l = np.hstack([L_ine, Beq, LB])
        u = np.hstack([B, Beq, UB])
        P = sparse.csc_matrix(sigMat)
        A = sparse.csc_matrix(A)

        prob = osqp.OSQP()
        # Setup workspace
        prob.setup(P, -meanVec, A, l, u, verbose=False)
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
                np.hstack([np.ones((1, d)), np.zeros((1, d)), np.zeros((1, d))]),
            ]
        )
        Beq = np.hstack([np.zeros(d), 1])
        LB = np.hstack([-Grenze * np.ones(d), np.zeros(2 * d)])
        UB = maxAlloc * np.ones(3 * d)
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
        prob.setup(sigMat3d, -meanvec3d, A, l, u, verbose=False)
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


def reinforcement_learning_portfolio(
    meanVec,
    sigMat,
    invHorizon = 10,
    initialWealth = 100,
    wealthGoal = 200,
    timeStep = 1,
    numPortfolios = 15,
    gridGranularity = 10,
    retFullStrat = False,
    hParams = None
):
    """
    This function uses the reinforcement learning (RL) approach described in Das & Varma (2020)
    to choose a portfolio in the efficient frontier for every time step

    :param meanVec: numpy.array
        vector of expected returns for each asset
    :param sigMat: numpy.array
        covariance matrix for the asset returns
    :param invHorizon: int
        investment horizon (in years) for the RL problem
    :param initialWealth: float
        initial wealth for the RL problem
    :param wealthGoal: float
        target wealth for the reward function of the RL problem
    :param timeStep: float
        time step ((dt) in years) for the each decision made before getting to the time horizon
    :param numPortfolios: int
        number of portfolios to choose from in the efficient frontier
    :param gridGranularity: int
        number of state values - 1 considered at each time step
    :param retFullStrat: Bool
        if True it will return the full strategy (np.array representing a tensor) for all time steps and states
    :param hParams: dict
        hyper-parameters used to train the RL strategy. See Das & Varma for more details
    :return: numpy.array
        vector of weights (% allocations) for each asset in the portfolio
    """


    if hParams is None:
        hParams = {
            "epsilon" : 0.3,
            "alpha" : 0.1,
            "gamma" : 1,
            "epochs" : 20000
        }

    # This portion of the code selects equally spaced portfolios in the efficient frontier
    # the "+ 5" is only to replicate the paper but no really necessary

    numPortGrid = numPortfolios + 5
    minMu = max(min(meanVec),0)
    maxMu = max(meanVec)
    # This is a hard-code to replicate the exact results of the paper
    # maxMu = 0.0989
    mu = np.linspace(minMu, maxMu, numPortGrid)

    weights = []
    sigma = []
    for mu_i in mu:
        w, var_i = meanVariancePortfolioReturnsTarget(meanVec,sigMat,mu_i,0)
        weights.append(w)
        sigma.append(np.sqrt(var_i))


    portfolios = [p for p in zip(mu.squeeze(), sigma.squeeze())]
    portfolios = np.array(portfolios[1:1 + numPortfolios])

    # Generating possible states following a geometric brownian motion

    gridPoints = invHorizon * gridGranularity + 1

    lnW = np.log(initialWealth)
    lnwMin = lnW
    lnwMax = lnW
    I = np.zeros((invHorizon))

    for t in range(invHorizon):
        lnwMin = np.log(np.exp(lnwMin) + I[t]) + (min(portfolios[:, 0]) - 0.5 * max(portfolios[:, 1]) ** 2) * timeStep - 3 * max(portfolios[:, 1]) * np.sqrt(timeStep)
        lnwMax = np.log(np.exp(lnwMax) + I[t]) + (max(portfolios[:, 0]) - 0.5 * max(portfolios[:, 1]) ** 2) * timeStep + 3 * max(portfolios[:, 1]) * np.sqrt(timeStep)
    W = np.exp(np.linspace(lnwMin, lnwMax, gridPoints)).squeeze()

    maxVec = np.vectorize(lambda x: 1 if x >= wealthGoal else 0)

    Q = np.zeros((gridPoints, invHorizon + 1, numPortfolios))
    R = np.zeros((gridPoints, invHorizon + 1, numPortfolios))
    R[:, invHorizon, :] = np.broadcast_to(np.array([maxVec(W)]).T, (gridPoints, numPortfolios))

    for e in range(hParams["epochs"]):
        Q = rl_update_policy_path(Q, R, invHorizon, timeStep, initialWealth, wealthGoal, portfolios, hParams)


    if retFullStrat:
        return weights, Q, W
    else:
        s_i = (W - initialWealth).argmin()
        return weights[Q[s_i, 0, :].argmax()], None, None


def rl_get_next_state(s_t, s_t1, port_i, portfolios, dt):
    """
        Based on a given state of the world (s_t) and an action (port_i) this function returns
        the next (random) state of the world

        Inputs:
        s_t is a scalar for S at t
        s_t1 is a vector of values for S at t1
        port_i is an integer for the index of the portfolio to be used

        Outputs:
        integer describing the index of the next state in the vector s_t1

    """
    mu = portfolios[port_i][0]
    sigma = portfolios[port_i][1]
    p1 = norm.pdf((np.log(s_t1 / s_t) - (mu - 0.5 * sigma ** 2) * dt) / (sigma * np.sqrt(dt)))
    p1 = p1 / sum(p1)
    idx = np.where(np.random.rand() > p1.cumsum())[0]
    return len(idx)


def rl_update_policy_node(i_w, t, Q, R, dt, T, W_0, W, portfolios, hParams):
    """
        This function is the core of the Q-learning algorithm. Given the index for a state and a time
        perform the "Q-update" of the Q-matrix


        Inputs:
        i_w is an integer index for S at t
        t is an integer index the moment in time

        Outputs:
        integer describing the index of the next state in the vector of space state

    """

    num_portfolios = len(portfolios)
    epsilon = hParams["epsilon"]
    alpha = hParams["alpha"]
    gamma = hParams["gamma"]


    # i_w: index on the wealth axis, t: index on the time axis
    # Pick optimal action a0 using epsilon greedy approach
    if np.random.rand() < epsilon:
        a = np.random.randint(0, num_portfolios)  # index of action; or plug in best action from last step
    else:
        q = Q[i_w, t, :]
        a = np.where(q == q.max())[0]  # Choose optimal Behavior policy
        if len(a) > 1:
            a = np.random.choice(a)  # randint(0,NP) #pick randomly from multiple maximizing actions
        else:
            a = a[0]

    # Generate next state S’ at t+1, given S at t and action a0, and update State -Action Value Function Q(S,A)
    t1 = t + 1
    if t < T:  # at t<T
        if t == 0:
            w0 = W_0
        else:
            w0 = W[i_w]  # scalar
        w1 = W  # vector
        i_w1 = rl_get_next_state(w0, w1, a, portfolios, dt)  # Model-free transition
        Q[i_w, t, a] = Q[i_w, t, a] + alpha * (R[i_w, t, a] + gamma * Q[i_w1, t1, :].max() - Q[i_w, t, a])  # THIS IS Q-LEARNING
    else:  # at T
        Q[i_w, t, a] = (1 - alpha) * Q[i_w, t, a] + alpha * R[i_w, t, a]
        i_w1 = i_w

    return i_w1, Q  # gives back next state (index of W and t)


def rl_update_policy_path(Q, R, T, dt, W_0, W, num_portfolios, hParams):
    """
        Run policy node for all time steps until T
    """
    i_w = 0
    for t in range(T+1):
        i_w, Q = rl_update_policy_node(i_w, t, Q, R, T, dt, W_0, W, num_portfolios, hParams)

    return Q


def check_missing(df_logret):
    """
    function to check the missing values and delete the stocks with missing value

    Parameters
    ----------
    df_logret : pandas.core.frame.DataFrame
       the price window

    Returns
    -------
    res : pandas.core.frame.DataFrame
       the price window without missing value
    """
    df_logret = df_logret.transpose()
    flag = np.zeros(len(df_logret))
    for i in range(len(df_logret)):
        if df_logret.iloc[i, :].isnull().any():
            flag[i] = 0
        else:
            flag[i] = 1
    df_logret["missing_flag"] = flag
    res = df_logret.loc[df_logret["missing_flag"] == 1]
    return res.transpose()


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
):
    """
    function do the rolling window back test

    Parameters
    ----------
    optimizerName : String
        The name of the optimizer to use for rolling window exercise
    data : Dictionary
        Data with Ticker, Date and Adjusted Close price
    whindow_size : int
        parameter for the size of rolling window
    rebalance_time : int
        rebalance time of rolling window test
    maxAlloc : Float
        maximum allocation. Takes values between 0 and 1
    riskAversion : Float
        Riske Aversion for your portfolio. Takes values greater than 0
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

    Returns
    -------
    R : 2d array
        return matrix depends on the rebalance time
    logret: 2d array
        log return matrix for each stocks
    w_all: 2d array
        optimal weight for each revalance time
    rownames: array
        date time of rolling window test

    Notes
    -------
    Note for now we have provided additional parameters that'll be used in future versions of the optimizers
    """
    df = pd.DataFrame(data)

    df.columns = ["date", "ticker", "price"]
    df1 = df.pivot_table(index=["date"], columns="ticker", values=["price"])
    df1.columns = [col[1] for col in df1.columns.values]
    df_logret = 100 * (np.log(df1) - np.log(df1.shift(1)))
    df_logret = df_logret[1:]
    logret = np.array(df_logret)
    n = logret.shape[0]
    d = rebalance_time
    start = window_size
    R = None
    portfolio_return = None
    w_all = None
    for i in range(start, n, d):
        k = 0
        w_opt = np.zeros(df1.shape[1])
        # import pdb; pdb.set_trace()
        window = check_missing(df_logret[i - window_size : i] / 100)
        m = window.shape[0]
        sample_stocks = window.columns
        logret_window = np.array(window.iloc[: n - 1])
        sigMat = np.cov(logret_window, rowvar=False)
        meanVec = np.mean(logret_window, axis=0) / 100

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

        for j in range(df1.shape[1]):
            if df1.columns[j] in sample_stocks:
                w_opt[j] = w_sample[k]
                k += 1

        if w_all is None:
            w_all = w_opt
        else:
            w_all = np.vstack([w_all, w_opt])

        if (i + d) < n:
            if R is None:
                logret_sample = np.nan_to_num(logret[i : i + d], nan=0)
                simple_returns = 100 * (math.exp(1) ** (logret_sample / 100) - 1)
                R = np.dot(w_opt, simple_returns.transpose())
            else:
                logret_sample = np.nan_to_num(logret[i : i + d], nan=0)
                simple_returns = 100 * (math.exp(1) ** (logret_sample / 100) - 1)
                R = np.hstack([R, np.dot(w_opt, simple_returns.transpose())])
        elif (i + d) >= n:
            logret_sample = np.nan_to_num(logret[i:], nan=0)
            simple_returns = 100 * (math.exp(1) ** (logret_sample / 100) - 1)
            R = np.hstack([R, np.dot(w_opt, simple_returns.transpose())])
    rownames = df1.index[start + 1 :]
    return R, df_logret, w_all, rownames


if __name__ == "__main__":
    pass
