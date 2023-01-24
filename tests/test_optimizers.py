from PyPortOpt import Optimizers as o
import unittest
import numpy as np
import pandas as pd
import logging
from pathlib import Path

##################
#  LOGGING SETUP #
##################
debug = False

if debug:
    logMode = logging.DEBUG
else:
    logMode = logging.INFO

# create logger
logger = logging.getLogger("tests")
# set log level for all handlers to debug
logger.setLevel(logMode)

# create console handler and set level to debug
# best for development or debugging
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logMode)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
consoleHandler.setFormatter(formatter)

# add ch to logger
logger.addHandler(consoleHandler)
######################
#  END LOGGING SETUP #
######################


class TestOptimizer(unittest.TestCase):
    def test_testFunction(self):
        self.assertEqual(o.testFunction(), True)

    def test_preprocessData(self):
        data = {
            "Ticker": {
                0: "AAPL",
                1: "AAPL",
                2: "AAPL",
                3: "AAPL",
                4: "AAPL",
                5: "AAPL",
                6: "AAPL",
                7: "TSLA",
                8: "TSLA",
                9: "TSLA",
                10: "TSLA",
                11: "TSLA",
                12: "TSLA",
                13: "TSLA",
            },
            "Date": {
                0: "2020-01-02",
                1: "2020-01-03",
                2: "2020-01-06",
                3: "2020-01-07",
                4: "2020-01-08",
                5: "2020-01-09",
                6: "2020-01-10",
                7: "2020-01-02",
                8: "2020-01-03",
                9: "2020-01-06",
                10: "2020-01-07",
                11: "2020-01-08",
                12: "2020-01-09",
                13: "2020-01-10",
            },
            "Adjusted_Close": {
                0: 74.09522915781685,
                1: 73.37487600602452,
                2: 73.95954620114364,
                3: 73.61170443949048,
                4: 74.79584660682033,
                5: 76.38457068132122,
                6: 76.55725808072349,
                7: 86.052,
                8: 88.602,
                9: 90.308,
                10: 93.812,
                11: 98.428,
                12: 96.268,
                13: 95.63,
            },
        }

        meanVec, sigMat, df_logret = o.preprocessData(data)

        self.assertEqual(meanVec.shape[0], 2)

        self.assertEqual(sigMat.shape[0], 2)

        self.assertEqual(sigMat.shape[1], 2)

    def test_SymPDcovmatrix(self):
        a = [[1, 2, 3], [5, 6, 7], [3, 5, 9]]
        a = np.array(a)
        SPD = np.dot(a.T, a)
        nonSPD = a

        mat, _ = o.SymPDcovmatrix(SPD, tol=1e-8)
        self.assertTrue(np.allclose(mat, SPD, atol=1e-8))

        mat, _ = o.SymPDcovmatrix(nonSPD, tol=1e-8)
        eig, _ = np.linalg.eig(mat)
        self.assertTrue(np.any(eig > 0))

    def test_sigMatShrinkage(self):
        a = [[1, 0, 0], [0, 3, 0], [0, 0, 4]]
        a = np.array(a)
        l2 = 0.7
        c = a + l2 * np.mean(np.diag(a)) * np.eye(3)

        b = o.sigMatShrinkage(a, l2)
        self.assertTrue(np.allclose(b, c))

    def test_Dmat(self):

        n = 3

        k1 = np.eye(3)
        k2 = -1.0 * np.ones((3, 3))
        k2 = np.triu(np.tril(k2, 1))
        np.fill_diagonal(k2, 0)
        k2 = k1 + k2
        k2 = k2[:2, :]

        self.assertTrue(np.allclose(o.Dmat(n, 0), k1))

        self.assertTrue(np.allclose(o.Dmat(n, 1), k2))

    def test_portfolio_optimization(self):
        data = {
            "Date": {
                0: "2020-01-02",
                1: "2020-01-03",
                2: "2020-01-06",
                3: "2020-01-07",
                4: "2020-01-08",
                5: "2020-01-09",
                6: "2020-01-10",
                7: "2020-01-02",
                8: "2020-01-03",
                9: "2020-01-06",
                10: "2020-01-07",
                11: "2020-01-08",
                12: "2020-01-09",
                13: "2020-01-10",
            },
            "Ticker": {
                0: "AAPL",
                1: "AAPL",
                2: "AAPL",
                3: "AAPL",
                4: "AAPL",
                5: "AAPL",
                6: "AAPL",
                7: "TSLA",
                8: "TSLA",
                9: "TSLA",
                10: "TSLA",
                11: "TSLA",
                12: "TSLA",
                13: "TSLA",
            },
            "Adjusted_Close": {
                0: 74.09522915781685,
                1: 73.37487600602452,
                2: 73.95954620114364,
                3: 73.61170443949048,
                4: 74.79584660682033,
                5: 76.38457068132122,
                6: 76.55725808072349,
                7: 86.052,
                8: 88.602,
                9: 90.308,
                10: 93.812,
                11: 98.428,
                12: 96.268,
                13: 95.63,
            },
        }
        meanVec, sigMat, df_logret = o.preprocessData(data)

        w_opt, var_opt = portfolio_optimization(meanVec,sigMat,retTarget = 0,longShort = 1,maxAlloc=1,lambda_l1=0,lambda_l2=0,riskfree = 0,assetsOrder=None,maxShar = 0,
        turnover = None, w_pre = None, individual = False, exposure_constrain = 0, w_bench = None, factor_exposure_constrain = None, U_factor = None, 
        general_linear_constrain = None, U_genlinear = 0, w_general = None, TE_constrain = 0, general_quad = 0, Q_w = None, Q_b = None, Q_bench = None)
        logger.debug(w_opt)
        w_opt_act = np.array([0.76501967, 0.23498033])
        var_opt_act = 0.7938368338395924

        self.assertTrue(np.allclose(w_opt, w_opt_act, atol=1e-8))

        self.assertTrue(np.allclose(var_opt, var_opt_act, atol=1e-8))


    def test_dynamic_programming_portfolio(self):
        homedir = Path(__name__)
        try:
            data_df = pd.read_parquet("./tests/index_data.parquet")
        except FileNotFoundError:
            data_df = pd.read_parquet(str(homedir.parent / "index_data.parquet"))
        meanVec, sigMat, df_logret = o.preprocessData(data_df.dropna(how='all').iloc[:504, :20])
        meanVec = np.expand_dims(meanVec/100, axis=1)
        dpStrat, dpV = o.dynamic_programming_portfolio(
                meanVec,
                sigMat/10000,
                shrinkage=0,
                timeStep=1,
                numPortfolios=15,
                wealthGoal=200,
                cashInjection=10,
        )

        logger.debug(dpV[0, 0])
        self.assertAlmostEqual(dpV[0, 0], 0.694515191511611, 5)


    def test_q_learning(self):
        homedir = Path(__name__)
        try:
            data_df = pd.read_parquet("./tests/index_data.parquet")
        except FileNotFoundError:
            data_df = pd.read_parquet(str(homedir.parent / "index_data.parquet"))
        meanVec, sigMat, df_logret = o.preprocessData(data_df.dropna(how='all').iloc[:501, :10])
        meanVec = np.expand_dims(meanVec/100, axis=1)
        hparams = dict(epsilon=0.3, alpha=0.1, gamma=0.9, epochs=10000)
        dpStrat, dpV = o.q_learning_portfolio(
                meanVec,
                sigMat/10000,
                shrinkage=0,
                invHorizon=10,
                timeStep=1,
                numPortfolios=15,
                wealthGoal=200,
                cashInjection=10,
                hParams=hparams
        )
        logger.debug(dpV[0, 0].mean())
        self.assertAlmostEqual(dpV[0, 0].mean(), 0.13993675763327576, 1)

    def test_g_learning(self):
        homedir = Path(__name__)
        try:
            data_df = pd.read_parquet("./tests/index_data.parquet")
        except FileNotFoundError:
            data_df = pd.read_parquet(str(homedir.parent / "index_data.parquet"))
        meanVec, sigMat, logret = o.preprocessData(data_df.iloc[:15, :10].dropna('columns', how='any'))
        # meanVec = np.expand_dims(meanVec/100, axis=1)
        logret = logret/100

        n, m = logret.shape
        d = 4
        window_size = 3
        start = window_size
        g_learner = o.g_learn(
            num_steps=20, rebalance_time=d, num_risky_assets=logret.shape[1],
            x_vals_init=1000*np.ones(logret.shape[1]) / logret.shape[1],
            lambd=0.001, omega=1.0, eta=1.5, rho=0.4,
            beta=1000.0, gamma=0.95, target_return=0.8
        )
        np.random.seed(2022)
        w = np.empty((0,m), float)
        for i in range(start, n, d):
            logret_window = logret[i-window_size:i]
            sigMat = np.cov(logret_window, rowvar=False)
            meanVec = np.mean(logret_window, axis=0)
            if i+d <= n:
                w_opt, g_learner = o.g_learn_rolling(
                    t=int((i-start)/d % g_learner.num_steps), g_learner=g_learner,
                    exp_returns=meanVec*d, sigma=sigMat*d,
                    returns=logret.iloc[i:i+d].sum(axis=0).values
                )
            else:
                d_final = n - i
                w_opt, g_learner = o.g_learn_rolling(
                    t=int((i-start)/d % g_learner.num_steps), g_learner=g_learner,
                    exp_returns=meanVec*d_final, sigma=sigMat*d_final,
                    returns=logret.iloc[i:i+d].sum(axis=0).values
                )
            w = np.r_[w, np.reshape(w_opt, (-1,m))]

        self.assertAlmostEqual(w_opt[0], 0.10768308253628003, 6)


    def test_rollingWindow(self):
        data = {
            "Date": {
                0: "2020-01-02",
                1: "2020-01-03",
                2: "2020-01-06",
                3: "2020-01-07",
                4: "2020-01-08",
                5: "2020-01-09",
                6: "2020-01-10",
                7: "2020-01-02",
                8: "2020-01-03",
                9: "2020-01-06",
                10: "2020-01-07",
                11: "2020-01-08",
                12: "2020-01-09",
                13: "2020-01-10",
            },
            "Ticker": {
                0: "AAPL",
                1: "AAPL",
                2: "AAPL",
                3: "AAPL",
                4: "AAPL",
                5: "AAPL",
                6: "AAPL",
                7: "TSLA",
                8: "TSLA",
                9: "TSLA",
                10: "TSLA",
                11: "TSLA",
                12: "TSLA",
                13: "TSLA",
            },
            "Adjusted_Close": {
                0: 74.09522915781685,
                1: 73.37487600602452,
                2: 73.95954620114364,
                3: 73.61170443949048,
                4: 74.79584660682033,
                5: 76.38457068132122,
                6: 76.55725808072349,
                7: 86.052,
                8: 88.602,
                9: 90.308,
                10: 93.812,
                11: 98.428,
                12: 96.268,
                13: 95.63,
            },
        }

        R, logRet, w, rownames = rollingwindow_backtest(
            "minimumVariancePortfolio", data, 2, 1
        )

        # logger.debug(R)

        R_act = [2.29679052, 2.93258696, -2.19449754, 0.16633527]

        logRet_act = [
            [-0.97695581, 2.9202666 ],
            [ 0.79366825, 1.90716194],
            [-0.471423, 3.80667295],
            [ 1.5958316, 4.80325368],
            [ 2.10183646, -2.21893478],
            [ 0.22582112, -0.66493903]
        ]
        w_act = [[ 3.63938000e-01,  6.36062001e-01],
            [ 6.00236889e-01,  3.99763111e-01],
            [-9.14856000e-09,  1.00000001e+00],
            [ 9.32785357e-01,  6.72146510e-02]]

        self.assertTrue(np.allclose(R, R_act, atol=1e-8))

        self.assertTrue(np.allclose(logRet.to_numpy(), logRet_act, atol=1e-8))
        self.assertTrue(np.allclose(w, w_act, atol=1e-8))
        self.assertEqual(len(R), len(rownames))
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
