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
    """
    This function calculates P(s_t1 | st) according the mean (mu) and volatility (sigma) of a normal distribution.

    Inputs:
    s_t is a scalar
    s_t1 is a vector
    dist is an array of [mean, volatility]

    Outputs:
    array that describes the pmf of P(s_t1 | st)
    """
    mu = dist[0]
    sigma = dist[1]
    p1 = norm.pdf(
        (np.log(s_t1 / s_t) - (mu - 0.5 * sigma ** 2) * dt) / (sigma * np.sqrt(dt))
    )
    return p1 / sum(p1)


def _expected_value(s_t, s_t1, v_t1, dt, dist):
    """
        This functions calculates the expected value of v_t1 based on P(s_t1 | st)

        Inputs:
        s_t is a scalar for S at t
        s_t1 is a vector of values for S at t1
        v_t1 is a vector of values for V at t1
        dist is an array of [mean, volatility] describing a normal distribution


        Outputs:
        scalar with  E_{P(s_t1 | st)}[v_t1]
    """
    return _transition_probabilities(s_t, s_t1, dt, dist).dot(v_t1)


def _create_wealth_grid(initialWealth, cashInjection, invHorizon, gridGranularity, dt, portfolios):
    """
    Creates wealth grid equally-spaced in log-space

    Parameters:
    -----------
    initialWealth: Float
        Starting wealth
    cashInjection: Float
        Periodic cash injection
    invHorizon: Integer
        Investment horizon in number of years
    gridGranularity: Integer
        Number of wealth values to generate per each year. See Das & Varma for more details
    dt: Float
        Time step as year fraction
    portfolios: numpy.array
        Array of portfolios. Each portfolio represented by mean return and volatility

    Returns:
    --------
    W: numpy.array
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

    Parameters:
    ----------
    wealthGrid: numpy.array
        Array with possible levels of wealth
    strategy: numpy.array
        Tensor that describes the value function of a RL problem. See q_learning_portfolio function
        for more details
    portfolios:
        Array of possible portfolios

    Returns:
    --------
    weight_function: Function
        See local weight_function documentation

    """

    def weight_function(currentWealth, t):
        """
        Parameters:
        ----------
        currentWealth: Float
            Current level of wealth
        t: Integer
            Index of number of time steps representing moment in time

        Returns:
        --------
        weight_function: Function

        """
        portIndex = np.abs(wealthGrid - currentWealth).argmin()
        if isinstance(portIndex, np.ndarray):
            portIndex = portIndex.max()
        return portfolios[strategy[portIndex, t]]

    return weight_function



