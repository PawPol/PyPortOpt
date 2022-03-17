from PyPortOpt import Optimizers as o
import numpy as np
import os
import pandas as pd
from pathlib import Path

data_df = pd.read_parquet(str(Path(os.getcwd()).parent.parent)+"/tests/index_data.parquet")
data_df = data_df.dropna(axis=0, how='all').dropna(axis=1, how='any').iloc[:, :10]
logret = o.preprocessData(data_df)[-1]
logret = logret/100

n = data_df.shape[0]
d = 21  # Rebalance time
window_size = 21*3  # Window size to estimate return
start = window_size  # Time to start investment
model_steps = 3  # The RL model steps is 3 (in month). After 3 months, we define a new RL model
currentWealth = 1000  # Initial cash

for rebalCount, i in enumerate(range(start, start+2*model_steps*d, d)):
    window = logret[i-window_size:i].copy()
    logret_window = window.values
    sigMat = np.cov(logret_window, rowvar=False)
    meanVec = np.mean(logret_window, axis=0)

    if i == start:
        g_learner = o.g_learn(
            num_steps=model_steps, num_risky_assets=logret.shape[1],
            x_vals_init=currentWealth*np.ones(logret.shape[1]) / logret.shape[1]
        )

    # w_sample, g_learner = o.g_learn_rolling(
    #     t=int((i-start)/d % g_learner.num_steps), g_learner=g_learner,
    #     exp_returns=meanVec*d, sigma=sigMat*d,
    #     returns=logret.iloc[i:i+d].sum(axis=0).values
    # )

    t = int((i-start)/d % g_learner.num_steps)  # t = 0,1,2,...,11 when model_steps = 12
    returns = logret.iloc[i:i+d].sum(axis=0).values
    print(f"Information of {rebalCount} on day {i}:")
    print(f"We are at the {t+1} step of the model.")

    if t == 0:
        g_learner.update_before_step_1(t)  # only when t=0
        # Initial cash assigned to each asset
        print("Initial cash assigned to each asset")
        print(g_learner.x_vals_init)
        # Initial value of portfolio-independent benchmark
        print("Initial value of portfolio-independent benchmark")
        print(g_learner.benchmark_portf)

    # Cash for each asset now
    print("Cash for each asset now")
    print(g_learner.x_t)

    w_sample = g_learner.run(t, meanVec*d, sigMat*d)
    # Cash change (action) for each asset
    print("Cash change (action) for each asset")
    print(g_learner.u_t)
    # Cash for each asset after action
    print("Cash for each asset after action")
    print(g_learner.x_next)
    # Portfolio weight
    print("Portfolio weight")
    print(w_sample)

    g_learner.update(t, returns)
    # Actual cash for each asset after d days
    print(f"Actual cash for each asset after {d} days")
    print(g_learner.x_t)
    # Trajectory of the model
    print("Trajectory of the model")
    print(g_learner.trajs)
    # Trajectory of rolling model
    print("Trajectory of rolling model")
    print(g_learner.trajs_all)












