import os
import pandas as pd
from PyPortOpt import Optimizers as o
import utils as u
import json
import logging
import quantstats as qs

logging.basicConfig(filename="../portOpt.log",
                        filemode="a", level=logging.INFO)
logging.info("Starting jobs")

with open("./jobs.list", "r+") as f:
    text = f.readlines()
    f.seek(0)
    f.truncate()

for job in text:
    
    job = job.replace("/n","")
    job = json.loads(job)
    email = job['email']
    print(email)
    optimizerName = job['Optimizer']
    tickers = job['Tickers'].upper().split(',')
    start_date = job['start_date']
    end_date = job['end_date']
    longShort = job['longShort']
    benchmark = job['benchmark'] 
    retTarget = job['retTarget']
    rollingWindowBackTest = job['rollingWindowBackTest']
    window_size = job['windowSize']
    rebalance_time = job['rebalanceTime']
    lambda_l2 = job['lambda_l2'] if 'lambda_l2' in job else ""
    initialWealth = job['initialWealth'] if 'initialWealth' in job else ""
    wealthGoal = job['wealthGoal'] if 'wealthGoal' in job else ""
    invHorizon = job['invHorizon'] if 'invHorizon' in job else ""
    stratUpdateFreq = job['stratUpdateFreq'] if 'stratUpdateFreq' in job else ""
    numPortOpt = job['numPortOpt'] if 'numPortOpt' in job else ""
    alpha = job['alpha'] if 'alpha' in job and job['alpha'] != '' else 0.1
    epsilon = job['epsilon'] if ('epsilon' in job) and job['epsilon'] != '' else 0.3
    gamma = job['gamma'] if ('gamma' in job) and job['gamma'] != '' else 1
    epochs = job['epochs'] if 'epochs' in job and job['epochs'] != '' else 100
    hParams = {
            "epsilon" : float(epsilon),
            "alpha" : float(alpha),
            "gamma" : float(gamma),
            "epochs" : int(epochs)
        }

    gLearningStep = job['gLearningStep'] if 'gLearningStep' in job else 12
    gLearningGamma = job['gLearningGamma'] if 'gLearningGamma' in job else 0.95
    gLearningBeta = job['gLearningBeta'] if 'gLearningBeta' in job else 1000
    gLearningTarget = job['gLearningTarget'] if 'gLearningTarget' in job else 0.8
    gLearningCash = job['gLearningCash'] if 'gLearningCash' in job else 1000
    gLearningLambda = job['gLearningLambda'] if 'gLearningLambda' in job else 0.001
    gLearningOmega = job['gLearningOmega'] if 'gLearningOmega' in job else 1.0
    gLearningEta = job['gLearningEta'] if 'gLearningEta' in job else 1.5
    gLearningRho = job['gLearningRho'] if 'gLearningRho' in job else 0.4

    idx = int(benchmark)
    benchmarkTicker = [tickers[idx-1]]
    tickers = tickers[:idx-1] + tickers[idx:]

    # generating query
    query_data = {"Tickers": tickers,
                  "start_date": start_date,
                  "end_date": end_date}
    query = u.generate_query(query_data)

    # getting data
    df = u.get_data(query)

    # Getting benchmark Data
    q = {"Tickers": benchmarkTicker,
         "start_date": start_date,
         "end_date": end_date}
    benchmarkData = u.get_data(u.generate_query(q))

    res = {"optimizerName": optimizerName,
           "tickers": tickers,
           "query": query
           }

    # optimizing
    if rollingWindowBackTest == "True":
        
        df = pd.DataFrame(df)
        df.columns = ['ticker', 'date', 'price']
        df = df[["date", "ticker", "price"]]
        # df.to_csv("../data.csv", index = False)
        
        if optimizerName == "qLearning":
            optimizerName = "q_learning"
        if optimizerName == "dynamicProgramming":
            optimizerName = "dynamic_programming"
        if optimizerName == "gLearning":
            optimizerName = "G-learning"

        R, logret, w_all, rownames = o.rollingwindow_backtest(
            optimizerName, df.to_dict(), int(window_size), int(rebalance_time),
            initialWealth=float(gLearningCash),
            g_steps=int(gLearningStep),
            g_goal_rate=float(gLearningTarget),
            g_lambda=float(gLearningLambda),
            g_omega=float(gLearningOmega),
            g_eta=float(gLearningEta),
            g_rho=float(gLearningRho),
            g_beta=float(gLearningBeta),
            g_gamma=float(gLearningGamma)
        )
            
        res["R"] = R.flatten().tolist()
        res["logret"] = logret.to_dict()
        res["w_all"] = w_all.tolist()
        res["rownames"] = list(rownames)
        res["benchmarkData"] = benchmarkData

        R = res["R"]
        R = [float(el) for el in R]
        dates = res["rownames"]
        benchmarkData = pd.DataFrame(res["benchmarkData"])
        benchmarkData = benchmarkData[-1*len(dates)-1:]
        benchmarkData = benchmarkData["Adjusted_Close"].pct_change()
        benchmarkData = benchmarkData[1:]
        benchmarkData = benchmarkData.to_list()

        df =u.prepossessing_ret(R, dates)
        df1 =u.prepossessing_ret(benchmarkData, dates)
        returns = pd.Series(R, index = pd.to_datetime(dates))/100
        benchmark = pd.Series(benchmarkData, index = pd.to_datetime(dates))

        qs.reports.html(returns, benchmark, output = 'file.html')
        with open('./file', 'r') as f:
            html = f.read()
        print(html)
        # print(html)
        # metrics = qs.reports.metrics(returns, benchmark, Mode = 'full', display = False)

