import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import quantstats as qs


def prepossessing_ret(R, name):
    """
    function get a dataframe of portfolio returns

    Parameters
    ----------
    R : array
            portfolio returns
    name: pandas.core.indexes.base.Index
            date time of the rolling window

    Returns
    -------
    df_sub : pandas.core.frame.DataFrame
    """
    df_sub = pd.DataFrame(R, index=name)
    df_sub.index = pd.to_datetime(df_sub.index)
    df_sub = df_sub.rename(columns={0: "strategy"})
    return df_sub


def cumret_plot(df_sub):
    """
    function to plot portfolio cumulative returns

    Parameters
    ----------
    df_sub : pandas.core.frame.DataFrame

    """
    figure(figsize=(15, 6), dpi=80)
    plt.plot(df_sub.index, np.cumprod(1 + df_sub["strategy"] / 100), label="strategy")
    plt.title("Cumulative Returns")
    plt.legend(loc="best")


def cumret_log_plot(df_sub):
    figure(figsize=(15, 6), dpi=80)
    plt.plot(
        df_sub.index, np.log(np.cumprod(1 + df_sub["strategy"] / 100)), label="strategy"
    )
    plt.title("Cumulative Returns log scale")
    plt.legend(loc="best")


def drawdown_plot(df_sub):
    figure(figsize=(15, 6), dpi=80)
    y = qs.stats.to_drawdown_series(df_sub["strategy"] / 100)
    plt.plot(df_sub.index, y, label="strategy")
    plt.title("Drawdown plot")
    plt.legend(loc="best")


def rolling_vol_plt(df_sub, window_size=5, rebalance_time=1):
    figure(figsize=(15, 6), dpi=80)
    plt.plot(
        rollingwindow_stat(df_sub).index, rollingwindow_stat(df_sub), label="strategy"
    )
    plt.title("rolling volatility")
    plt.legend(loc="best")


def rolling_shar_plt(df_sub, window_size=5, rebalance_time=1):
    figure(figsize=(15, 6), dpi=80)
    plt.plot(
        rollingwindow_shar(df_sub).index, rollingwindow_shar(df_sub), label="strategy"
    )
    plt.title("rolling sharpe ratio")
    plt.legend(loc="best")


def rolling_sortino_plt(df_sub, window_size=5, rebalance_time=1):
    figure(figsize=(15, 6), dpi=80)
    plt.plot(
        rollingwindow_sortino(df_sub).index,
        rollingwindow_sortino(df_sub),
        label="strategy",
    )
    plt.title("rolling sortino")
    plt.legend(loc="best")


def rollingwindow_stat(df_sub, window_size=5, rebalance_time=1):

    """
    function get the rolling volativity

    Parameters
    ----------
    df_sub : pandas.core.frame.DataFrame
            portfolio returns
    whindow_size : int
        parameter for the size of rolling window
    rebalance_time : int
        rebalance time of rolling window test

    Returns
    -------
    vol_all : pandas.core.series.Series
            rolling volativity
    """
    port_ret = df_sub["strategy"]

    n = port_ret.shape[0]
    d = rebalance_time
    start = window_size
    R = None
    vol_all = None
    for i in range(start, n, d):
        if (i + d) < n:
            window = port_ret[i - window_size : i] / 100
            vol = np.std(window) * np.sqrt(252)

            if vol_all is None:
                vol_all = vol
            else:
                vol_all = np.hstack([vol_all, vol])
        elif (i + d) >= n:
            window = port_ret[i - window_size :] / 100
            vol = np.std(window) * np.sqrt(252)
            vol_all = np.hstack([vol_all, vol])
    vol_all = pd.Series(vol_all, index=port_ret[start:n].index)
    return vol_all


def rollingwindow_shar(df_sub, window_size=5, rebalance_time=1):
    """
    function get the rolling volativity

    Parameters
    ----------
    df_sub : pandas.core.frame.DataFrame
            portfolio returns
    whindow_size : int
        parameter for the size of rolling window
    rebalance_time : int
        rebalance time of rolling window test

    Returns
    -------
    shar_all : pandas.core.series.Series
            rolling sharp ratio
    """
    port_ret = df_sub["strategy"]

    n = port_ret.shape[0]
    d = rebalance_time
    start = window_size
    R = None
    shar_all = None
    for i in range(start, n, d):
        if (i + d) < n:
            window = port_ret[i - window_size : i] / 100
            shar = qs.stats.sharpe(window)

            if shar_all is None:
                shar_all = shar
            else:
                shar_all = np.hstack([shar_all, shar])
        elif (i + d) >= n:
            window = port_ret[i - window_size :] / 100
            shar = qs.stats.sharpe(window)
            shar_all = np.hstack([shar_all, shar])
    shar_all = pd.Series(shar_all, index=port_ret[start:n].index)
    return shar_all


def rollingwindow_sortino(df_sub, window_size=5, rebalance_time=1):
    """
    function get the rolling volativity

    Parameters
    ----------
    df_sub : pandas.core.frame.DataFrame
            portfolio returns
    window_size : int
        parameter for the size of rolling window
    rebalance_time : int
        rebalance time of rolling window test

    Returns
    -------
    shar_all : pandas.core.series.Series
            rolling sortino
    """
    port_ret = df_sub["strategy"]

    n = port_ret.shape[0]
    d = rebalance_time
    start = window_size
    R = None
    sortino_all = None
    for i in range(start, n, d):
        if (i + d) < n:
            window = port_ret[i - window_size : i] / 100
            sortino = qs.stats.sortino(window)

            if sortino_all is None:
                sortino_all = sortino
            else:
                sortino_all = np.hstack([sortino_all, sortino])
        if (i + d) >= n:
            window = port_ret[i - window_size :] / 100
            sortino = qs.stats.sortino(window)
            sortino_all = np.hstack([sortino_all, sortino])
    sortino_all = pd.Series(sortino_all, index=port_ret[start:n].index)
    return sortino_all
