import logging
import numpy as np
from scipy.stats import norm


##################
#  LOGGING SETUP #
##################
debug = False

if debug:
    logMode = logging.DEBUG
else:
    logMode = logging.INFO

# create logger
logger = logging.getLogger("reinforce")
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


def _get_next_state(s_t, s_t1, port_i, portfolios, dt, hist=None, weights=None):
    """
    Based on a given state of the world (s_t) and an action (port_i) this function returns
    the next (random) state of the world

    Parameters
    ----------
    s_t: float
        value of S at time t
    s_t1: array_like
        vector of values for S at t1
    port_i: int
        integer for the index of the portfolio to be used
    portfolios: list
        total possible portfolios
    dt: float
        time step
    hist: pandas.core.frame.DataFrame or None
        if None use gbm to generate next state if df sample from it
    weights: np.ndarray
        only used if hist is not None. weights of each asset in the portfolio

    Returns
    -------
    next_state_index: int
        integer describing the index of the next state in the vector s_t1

    """
    if hist is None:
        mu = portfolios[port_i][0]
        sigma = portfolios[port_i][1]
        p1 = norm.pdf(
            (np.log(s_t1 / s_t) - (mu - 0.5 * sigma ** 2) * dt) / (sigma * np.sqrt(dt))
        )
        p1 = p1 / sum(p1)
        idx = np.where(np.random.rand() > p1.cumsum())[0]
        next_state_index = len(idx)
    else:
        logger.debug(weights.sum())
        logger.debug(weights)
        logger.debug((hist.sample(int(dt * 252)).values @ weights).sum())
        s_t1_new = s_t * np.exp((hist.sample(int(dt * 252)).values @ weights).sum())
        next_state_index = np.abs(s_t1 - s_t1_new).argmin()

    return next_state_index


def _update_policy_node(
    i_w, t, Q, R, T, dt, W_0, W, portfolios, hParams, returns, weights
):
    """
    This function is the core of the Q-learning algorithm. Given the index for a state and a time
    perform the "Q-update" of the Q-matrix


    Parameters
    ----------
    i_w: int
        index for S at time t
    t: int
        index for the moment in time
    Q: numpy.ndarray
        Q-tensor
    R: numpy.ndarray
        R-tensor
    T: int
        time horizon
    dt: float
        time step
    W_0: float
        initial wealth
    W: array_like
        possible wealth values
    portfolios: array_like
        possible investment portfolios
    hParams: dict
        hyper-parameters for the model
    returns: pandas.core.frame.DataFrame
        DataFrame of historical daily returns for all assets
    weights: array_like
        weights of all assets for all portfolios

    Returns
    -------
    i_w1: int
        index of the next state in the vector of space state
    Q: numpy.ndarray
        Q-tensor

    """

    num_portfolios = len(portfolios)
    epsilon = hParams["epsilon"]
    alpha = hParams["alpha"]
    gamma = hParams["gamma"]

    # i_w: index on the wealth axis, t: index on the time axis
    # Pick optimal action a0 using epsilon greedy approach
    if np.random.rand() < epsilon:
        a = np.random.randint(
            0, num_portfolios
        )  # index of action; or plug in best action from last step
    else:
        q = Q[i_w, t, :]
        a = np.where(q == q.max())[0]  # Choose optimal Behavior policy
        if len(a) > 1:
            a = np.random.choice(
                a
            )  # randint(0,NP) #pick randomly from multiple maximizing actions
            # a = a.max()
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
        i_w1 = _get_next_state(
            w0, w1, a, portfolios, dt, returns, weights[a]
        )  # Model-free transition
        # logger.debug(i_w1)
        Q[i_w, t, a] = Q[i_w, t, a] + alpha * (
            R[i_w, t, a] + gamma * Q[i_w1, t1, :].max() - Q[i_w, t, a]
        )  # THIS IS Q-LEARNING
    else:  # at T
        Q[i_w, t, a] = (1 - alpha) * Q[i_w, t, a] + alpha * R[i_w, t, a]
        i_w1 = i_w

    return i_w1, Q  # gives back next state (index of W and t)


def update_policy_path(Q, R, T, dt, W_0, W, portfolios, hParams, returns, weights):
    """Run policy node for all time steps until T

    Parameters
    ----------
    Q: numpy.ndarray
        Q-tensor
    R: numpy.ndarray
        R-tensor
    T: int
        time horizon
    dt: float
        time step
    W_0: float
        initial wealth
    W: array_like
        possible wealth values
    portfolios: array_like
        possible investment portfolios
    hParams: dict
        hyper-parameters for the model
    returns: pandas.core.frame.DataFrame
        DataFrame of historical daily returns for all assets
    weights: array_like
        weights of all assets for all portfolios

    Returns
    -------
    Q: numpy.ndarray
        Q-tensor
    """
    i_w = 0
    for t in range(T + 1):
        i_w, Q = _update_policy_node(
            i_w, t, Q, R, T, dt, W_0, W, portfolios, hParams, returns, weights
        )

    return Q
