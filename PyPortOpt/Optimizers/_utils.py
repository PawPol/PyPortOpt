import numpy as np
from scipy.stats import norm


def _check_missing(df_logret):
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


def _transition_probabilities(s_t, s_t1, dt, dist):
    """Calculates conditional probability current price and a normal distribution

    Calculates P(s_t1 | st) according the mean (mu) and volatility (sigma) of a normal distribution.

    Parameters
    -----------
    s_t: float
        Price at time t
    s_t1: array_like
        Possible prices a t+1
    dist: array_like
        mean and volatility of the normal distribution

   Returns
   --------
    out: array_like
        pmf of s_t1 given st
    """
    mu = dist[0]
    sigma = dist[1]
    p1 = norm.pdf(
        (np.log(s_t1 / s_t) - (mu - 0.5 * sigma ** 2) * dt) / (sigma * np.sqrt(dt))
    )
    return p1 / sum(p1)


def _expected_value(s_t, s_t1, v_t1, dt, dist):
    """Calculates the expected value of v_t1 based on P(s_t1 | st)

    Parameters
    -----------
    s_t: float
        Price at time t
    s_t1: array_like
        Possible prices a t+1
    v_t1: array_like
        Possible values of the random variable
    dist: array_like
        mean and volatility of the normal distribution


   Returns
   --------
    out: float
        Expected value of v_t1 at time t given st
    """
    return _transition_probabilities(s_t, s_t1, dt, dist).dot(v_t1)


def _create_wealth_grid(initialWealth, cashInjection, invHorizon, gridGranularity, dt, portfolios):
    """Creates wealth grid equally-spaced in log-space

    Parameters
    ----------
    initialWealth: float
        Starting wealth
    cashInjection: float
        Periodic cash injection
    invHorizon: int
        Investment horizon in number of years
    gridGranularity: int
        Number of wealth values to generate per each year. See Das & Varma for more details
    dt: float
        Time step as year fraction
    portfolios: numpy.ndarray
        Array of portfolios. Each portfolio represented by mean return and volatility

    Returns
    -------
    W: numpy.ndarray
        Wealth grid
    """
    gridPoints = invHorizon * gridGranularity + 1

    lnW = np.log(initialWealth)
    lnwMin = lnW
    lnwMax = lnW

    I = np.ones((invHorizon))*cashInjection

    maxMu, maxSigma = portfolios.max(axis=0)
    minMu, minSigma = portfolios.min(axis=0)

    for t in range(invHorizon):
        lnwMin = (
            np.log(np.exp(lnwMin) + I[t])
            + (minMu - 0.5 * maxSigma ** 2) * dt
            - 3 * maxSigma * np.sqrt(dt)
        )
        lnwMax = (
            np.log(np.exp(lnwMax) + I[t])
            + (maxMu - 0.5 * maxSigma ** 2) * dt
            + 3 * maxSigma * np.sqrt(dt)
        )
    W = np.exp(np.linspace(lnwMin, lnwMax, gridPoints)).squeeze()

    return W


def _create_weight_function(wealthGrid, strategy, portfolios):
    """
    Takes possible levels of wealth, possible portfolios, and a Q-tensor and turns them
    into a function

    Parameters
    ----------
    wealthGrid: numpy.ndarray
        Array with possible levels of wealth
    strategy: numpy.ndarray
        Tensor that describes the value function of a RL problem. See q_learning_portfolio function
        for more details
    portfolios:
        Array of possible portfolios

    Returns
    -------
    weight_function: function
        See local weight_function documentation

    """

    def weight_function(currentWealth, t):
        """
        Parameters
        ----------
        currentWealth: float
            Current level of wealth
        t: int
            Index of number of time steps representing moment in time

        Returns
        -------
        out: list
            list that represents portfolio, mean return and volatility
        """
        portIndex = np.abs(wealthGrid - currentWealth).argmin()
        if isinstance(portIndex, np.ndarray):
            portIndex = portIndex.max()
        return portfolios[strategy[portIndex, t]]

    return weight_function



