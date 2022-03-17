#%%
import numpy as np
import pandas as pd
import torch
import torch.optim as optim


# Define the G-learning portfolio optimization class
class G_learning_portfolio_opt:
    def __init__(self,
                 num_steps,
                 params,
                 beta,
                 benchmark_portf,
                 gamma,
                 num_risky_assets,
                 exp_returns,
                 Sigma_r,
                 init_x_vals,  # array of initial asset position values (num_risky_assets + 1)
                 use_for_WM = True):
        """
        Input several parameters into the model
        Parameters
        ----------
        num_steps : number of steps in RL model
        params : list containing reward function parameters lambda, omega, eta and rho
        beta : the parameter to determine the strength of entropy regularization
        benchmark_portf : the value of portfolio-independent benchmark
        gamma : discount factor in accumulative reward function
        num_risky_assets : number of risky assets in portfolio
        exp_returns : expected returns of risky assets at each step (shape: num_steps x num_risky_assets)
        Sigma_r : covariance matrix of risky assets returns
        init_x_vals : initial cash invested in portfolio
        use_for_WM : use for wealth management tasks (True or False)
        """
        # self.num_steps: number of steps in RL model
        self.num_steps = num_steps
        # self.num_assets:
        # number of assets in portfolio, if there is an additional risk free asset, then num_risky_assets + 1
        self.num_assets = num_risky_assets

        # self.lambd: the parameter to determine the penalty strength of not reaching target portfolio value
        self.lambd = torch.tensor(params[0], requires_grad=False, dtype=torch.float64)
        # self.Omega_mat:
        # The parameter matrix in a convex function as the term approximating transaction costs in reward function
        self.Omega_mat = params[1] * torch.eye(self.num_assets, dtype=torch.float64)
        # self.eta: (>1) the parameter that defines the desired growth rate of the current portfolio
        self.eta = torch.tensor(params[2], requires_grad=False, dtype=torch.float64)
        # self.rho: a relative weight of the portfolio-independent and portfolio-dependent terms
        self.rho = torch.tensor(params[3], requires_grad=False, dtype=torch.float64)
        # self.beta: the "inverse-temperature" parameter determines the strength of entropy regularization
        self.beta = torch.tensor(beta, requires_grad=False, dtype=torch.float64)
        # self.gamma: discount factor in accumulative reward function
        self.gamma = gamma

        self.use_for_WM = use_for_WM  # use for wealth management tasks or not

        self.num_risky_assets = num_risky_assets

        assert exp_returns.shape[0] == self.num_steps
        assert Sigma_r.shape[0] == Sigma_r.shape[1]
        assert Sigma_r.shape[0] == num_risky_assets  # self.num_assets

        # If there is an additional risk free asset,
        # then self.Sigma_r_np is [0,zeros(1,len(Sigma_r)); zeros(len(Sigma_r),1), 0]
        self.Sigma_r_np = Sigma_r  # array of shape num_stocks x num_stocks

        self.reg_mat = 1e-3*torch.eye(self.num_assets, dtype=torch.float64)

        # arrays of returns for all assets including the risk-free asset
        # array of shape num_steps x (num_stocks + 1)
        self.exp_returns_np = exp_returns

        # make block-matrix Sigma_r_tilde with Sigma_r_tilde[0,0] = 0, and equity correlation matrix inside
        self.Sigma_r_tilde_np = self.Sigma_r_np

        # make Torch tensors
        self.exp_returns = torch.tensor(self.exp_returns_np, requires_grad=False, dtype=torch.float64)
        self.Sigma_r = torch.tensor(Sigma_r, requires_grad=False, dtype=torch.float64)
        self.Sigma_r_tilde = torch.tensor(self.Sigma_r_tilde_np, requires_grad=False, dtype=torch.float64)

        # self.benchmark_portf: the value of portfolio-independent benchmark
        self.benchmark_portf = torch.tensor(benchmark_portf, requires_grad=False, dtype=torch.float64)

        # asset holding values for all times. Initialize with initial values,
        # values for the future times will be expected values
        self.x_vals_np = np.zeros((self.num_steps, self.num_assets))
        self.x_vals_np[0,:] = init_x_vals

        # Torch tensor
        self.x_vals = torch.tensor(self.x_vals_np)

        # allocate memory for coefficients of R-, F- and G-functions
        self.F_xx = torch.zeros(self.num_steps, self.num_assets, self.num_assets, dtype=torch.float64,
                                requires_grad=False)
        self.F_x = torch.zeros(self.num_steps, self.num_assets, dtype=torch.float64,
                               requires_grad=False)
        self.F_0 = torch.zeros(self.num_steps, dtype=torch.float64, requires_grad=False)

        self.Q_xx = torch.zeros(self.num_steps, self.num_assets, self.num_assets, dtype=torch.float64,
                                requires_grad=False)
        self.Q_uu = torch.zeros(self.num_steps, self.num_assets, self.num_assets, dtype=torch.float64,
                                requires_grad=False)
        self.Q_ux = torch.zeros(self.num_steps, self.num_assets, self.num_assets, dtype=torch.float64,
                                requires_grad=False)
        self.Q_x = torch.zeros(self.num_steps, self.num_assets, dtype=torch.float64, requires_grad=False)
        self.Q_u = torch.zeros(self.num_steps, self.num_assets, dtype=torch.float64, requires_grad=False)
        self.Q_0 = torch.zeros(self.num_steps, dtype=torch.float64, requires_grad=False)

        self.R_xx = torch.zeros(self.num_steps, self.num_assets, self.num_assets, dtype=torch.float64,
                                requires_grad=False)
        self.R_uu = torch.zeros(self.num_steps, self.num_assets, self.num_assets, dtype=torch.float64,
                                requires_grad=False)
        self.R_ux = torch.zeros(self.num_steps, self.num_assets, self.num_assets, dtype=torch.float64,
                                requires_grad=False)
        self.R_x = torch.zeros(self.num_steps, self.num_assets, dtype=torch.float64, requires_grad=False)
        self.R_u = torch.zeros(self.num_steps, self.num_assets, dtype=torch.float64, requires_grad=False)
        self.R_0 = torch.zeros(self.num_steps, dtype=torch.float64, requires_grad=False)

        self.reset_prior_policy()

        # the list of adjustable model parameters:
        self.model_params = [self.lambd, self.beta, self.Omega_mat, self.eta]

        # expected cash installment for all steps
        self.expected_c_t = torch.zeros(self.num_steps, dtype=torch.float64)

        # realized values of the target portfolio
        self.realized_target_portf = np.zeros(self.num_steps, dtype=np.float64)

        # expected portfolio values for all times
        self.expected_portf_val = torch.zeros(self.num_steps, dtype=torch.float64)

        # the first value is the sum of initial position values
        self.expected_portf_val[0] = self.x_vals[0,:].sum()

    def reset_prior_policy(self):
        """
        reset the RL model prior policy parameters into 0
        """
        # initialize time-dependent parameters of prior policy
        self.u_bar_prior = torch.zeros(self.num_steps, self.num_assets, requires_grad=False,
                                       dtype=torch.float64)
        self.v_bar_prior = torch.zeros(self.num_steps, self.num_assets, self.num_assets, requires_grad=False,
                                       dtype=torch.float64)
        self.Sigma_prior = torch.zeros(self.num_steps, self.num_assets, self.num_assets, requires_grad=False,
                                       dtype=torch.float64)
        self.Sigma_prior_inv = torch.zeros(self.num_steps, self.num_assets, self.num_assets, requires_grad=False,
                                           dtype=torch.float64)

        # make each time elements of v_bar_prior and Sigma_prior proportional to the unit matrix
        for t in range(self.num_steps):
            self.v_bar_prior[t,:,:] = 0.1 * torch.eye(self.num_assets).clone()
            self.Sigma_prior[t,:,:] = 0.1 * torch.eye(self.num_assets).clone()
            self.Sigma_prior_inv[t,:,:] = 10.0 * torch.eye(self.num_assets).clone()  # np.linalg.inv(self.Sigma_prior[t,:,:])

    def reward_fun(self, t, x_vals, u_vals, exp_rets, lambd, Sigma_hat):
        """
        Compute reward by reward function
        Parameters
        ----------
        t : the t-th time step of RL model
        x_vals : cash in assets
        u_vals : cash change in assets
        exp_rets : expected returns of assets
        lambd : the parameter to determine the penalty strength of not reaching target portfolio value
        Sigma_hat : Sigma_r (covariance matrix) + (1+expected returns)(1+expected returns)^transpose

        Returns
        -------
        Computed reward (aux_1 + aux_2 + aux_3 + aux_4 + aux_5)
        """
        x_plus = x_vals + u_vals

        p_hat = self.rho.clone() * self.benchmark_portf[t] + (1-self.rho.clone())*self.eta.clone()*x_vals.sum()

        aux_1 = - self.lambd.clone() * p_hat**2
        aux_2 = - u_vals.sum()
        aux_3 = 2*self.lambd.clone() * p_hat * x_plus.dot(torch.ones(self.num_assets) + exp_rets)
        aux_4 = - self.lambd.clone() * x_plus.mm(Sigma_hat.mv(x_plus))
        aux_5 = - u_vals.mm(self.Omega_mat.clone().mv(u_vals))

        return aux_1 + aux_2 + aux_3 + aux_4 + aux_5

    def compute_reward_fun(self):
        """
        Compute terms including R_xx, R_ux, etc. for all steps in reward function
        """
        for t in range(0, self.num_steps):

            one_plus_exp_ret = torch.ones(self.num_assets, dtype=torch.float64) + self.exp_returns[t,:]
            benchmark_portf = self.benchmark_portf[t]
            Sigma_hat = self.Sigma_r_tilde + torch.ger(one_plus_exp_ret, one_plus_exp_ret)

            one_plus_exp_ret_by_one = torch.ger(one_plus_exp_ret, torch.ones(self.num_assets, dtype=torch.float64))
            one_plus_exp_ret_by_one_T = one_plus_exp_ret_by_one.t()
            one_one_T_mat = torch.ones(self.num_assets, self.num_assets)

            self.R_xx[t,:,:] = (-self.lambd.clone()*(self.eta.clone()**2)*(self.rho.clone()**2)*one_one_T_mat
                                + 2*self.lambd.clone()*self.eta.clone()*self.rho.clone()*one_plus_exp_ret_by_one
                                - self.lambd.clone()*Sigma_hat)

            self.R_ux[t,:,:] = (2*self.lambd.clone()*self.eta.clone()*self.rho.clone()*one_plus_exp_ret_by_one
                                - 2*self.lambd.clone()*Sigma_hat)

            self.R_uu[t,:,:] = - self.lambd.clone() * Sigma_hat - self.Omega_mat.clone()

            self.R_x[t,:] = (-2*self.lambd.clone()*self.eta.clone()*self.rho.clone()*(1-self.rho.clone())*benchmark_portf *
                             torch.ones(self.num_assets, dtype=torch.float64)
                             + 2*self.lambd.clone()*(1-self.rho.clone())*benchmark_portf * one_plus_exp_ret)

            self.R_u[t,:] = (2*self.lambd.clone()*(1-self.rho.clone())*benchmark_portf * one_plus_exp_ret
                             - torch.ones(self.num_assets, dtype=torch.float64))

            self.R_0[t] = - self.lambd.clone()*((1-self.rho.clone())**2) * (benchmark_portf**2)

    def project_cash_injections(self):
        """
        Compute the expected values of future asset positions, and the expected cash injection for future steps,
        as well as realized values of the target portfolio in terms of trained policy.
        """
        # this assumes that the policy has been trained
        for t in range(1, self.num_steps):  # the initial value is fixed
            # increment the previous x_t
            delta_x_t = self.u_bar_prior[t,:] + self.v_bar_prior[t,:,:].mv(self.x_vals[t-1,:])
            self.x_vals[t,:] = self.x_vals[t-1,:] + delta_x_t

            # grow using the expected return
            self.x_vals[t,:] = (torch.ones(self.num_assets) + self.exp_returns[t,:])*self.x_vals[t,:]

            # compute c_t
            self.expected_c_t[t] = delta_x_t.sum().data  # detach().numpy()

            # expected portfolio value for this step
            self.expected_portf_val[t] = self.x_vals[t,:].sum().data  # .detach().numpy()

    def set_terminal_conditions(self):
        """
        Set the terminal condition for the F-function (Free energy function). That is to say, compute F-function at
        terminal step in RL model.
        """

        # the auxiliary quantity to perform matrix calculations
        one_plus_exp_ret = torch.ones(self.num_assets, dtype=torch.float64) + self.exp_returns[-1,:]

        # Compute the reward function for all steps (only the last step is needed for this functions, while
        # values for other time steps will be used in other functions)
        self.compute_reward_fun()

        if self.use_for_WM:
            Sigma_hat = self.Sigma_r_tilde + torch.ger(one_plus_exp_ret, one_plus_exp_ret)
            Sigma_hat_inv = torch.inverse(Sigma_hat + self.reg_mat)

            Sigma_tilde = Sigma_hat + (1/self.lambd)*self.Omega_mat.clone()
            Sigma_tilde_inv = torch.inverse(Sigma_tilde + self.reg_mat)

            Sigma_hat_sigma_tilde = Sigma_hat.mm(Sigma_tilde)
            Sigma_tilde_inv_sig_hat = Sigma_tilde_inv.mm(Sigma_hat)
            Sigma_tilde_sigma_hat = Sigma_tilde.mm(Sigma_hat)

            Sigma_hat_Sigma_tilde_inv = Sigma_hat.mm(Sigma_tilde_inv)
            Sigma_3_plus_omega = self.lambd*Sigma_tilde_inv.mm(Sigma_hat_Sigma_tilde_inv) + self.Omega_mat.clone()

            one_plus_exp_ret_by_one = torch.ger(one_plus_exp_ret, torch.ones(self.num_assets, dtype=torch.float64))
            one_plus_exp_ret_by_one_T = one_plus_exp_ret_by_one.t()
            one_one_T_mat = torch.ones(self.num_assets, self.num_assets)

            Sigma_tilde_inv_t_R_ux = Sigma_tilde_inv.t().mm(self.R_ux[-1,:,:].clone())
            Sigma_tilde_inv_t_R_uu = Sigma_tilde_inv.t().mm(self.R_uu[-1,:,:].clone())
            Sigma_tilde_inv_t_R_u = Sigma_tilde_inv.t().mv(self.R_u[-1,:].clone())

            Sigma_tilde_inv_R_u = Sigma_tilde_inv.mv(self.R_u[-1,:].clone())
            Sigma_tilde_inv_R_ux = Sigma_tilde_inv.mm(self.R_ux[-1,:,:].clone())
            Sigma_tilde_inv_t_R_uu = Sigma_tilde_inv.mm(self.R_uu[-1,:,:].clone())

            # though the action at the last step is deterministic, we can feed
            # parameters of the prior with these values

            self.u_bar_prior[-1,:] = (1/(2 * self.lambd.clone())) * Sigma_tilde_inv.clone().mv(self.R_u[-1,:].clone())
            self.v_bar_prior[-1,:,:] = (1/(2 * self.lambd.clone())) * Sigma_tilde_inv.clone().mm(self.R_ux[-1,:,:].clone())

            # First compute the coefficients of the reward function F at the last step:
            # F_xx
            self.F_xx[-1,:,:] = (self.R_xx[-1,:,:].clone()
                                 + (1/(2*self.lambd.clone())) * self.R_ux[-1,:,:].clone().t().mm(Sigma_tilde_inv_t_R_ux)
                                 + (1/(4*self.lambd.clone()**2)) * self.R_ux[-1,:,:].clone().t().mm(
                        Sigma_tilde_inv_t_R_uu.clone().mm(Sigma_tilde_inv.clone().mm(self.R_ux[-1,:,:].clone())))
                                 )

            # F_x
            self.F_x[-1,:] = (self.R_x[-1,:].clone()
                              + (1/(self.lambd.clone())) * self.R_ux[-1,:,:].clone().t().mv(Sigma_tilde_inv_t_R_u.clone())
                              + (1/(2*self.lambd.clone()**2)) * self.R_ux[-1,:,:].clone().t().mv(
                        Sigma_tilde_inv_t_R_uu.clone().mv(Sigma_tilde_inv_R_u.clone()))
                              )

            # F_0
            self.F_0[-1] = (self.R_0[-1].clone()
                            + (1/(2*self.lambd.clone())) * self.R_u[-1,:].clone().dot(Sigma_tilde_inv_R_u.clone())
                            + (1/(4*self.lambd.clone()**2)) * self.R_u[-1,:].clone().dot(
                        Sigma_tilde_inv_t_R_uu.clone().mv(Sigma_tilde_inv_R_u.clone()))
                            )

            # for the Q-function at the last step:
            self.Q_xx[-1,:,:] = self.R_xx[-1,:,:].clone()
            self.Q_ux[-1,:,:] = self.R_ux[-1,:,:].clone()
            self.Q_uu[-1,:,:] = self.R_uu[-1,:,:].clone()
            self.Q_u[-1,:] = self.R_u[-1,:].clone()
            self.Q_x[-1,:] = self.R_x[-1,:].clone()
            self.Q_0[-1] = self.R_0[-1].clone()

    def G_learning(self, err_tol, max_iter):
        """
        Find the optimal policy for the time dependent policy. This will run backward from the terminal step to
        initial step by calling step_G_learning function at each step.
        This should start from step t = num_steps - 2 (i.e. from a step that is before the last one).
        Parameters
        ----------
        err_tol : the parameter error tolerance to input into step_G_learning function
        max_iter : the parameter maximum iteration to input into step_G_learning function
        """
        print('Doing G-learning, it may take a few seconds...')

        # set terminal conditions
        self.set_terminal_conditions()

        # allocate iteration numbers for all steps
        self.iter_counts = np.zeros(self.num_steps)

        # iterate over time steps backward
        for t in range(self.num_steps-2, -1, -1):
            self.step_G_learning(t, err_tol, max_iter)

    def step_G_learning(self, t, err_tol, max_iter):
        """
        Perform one step of backward iteration for G-learning self-consistent equations.
        Parameters
        ----------
        t : the t-th step of model that the function is running
        err_tol : error tolerance - once the updated u_bar and v_bar move less than error tolerance.
                  The function will stop iteration.
        max_iter : maximum iteration
        """

        # make matrix Sigma_hat_t
        one_plus_exp_ret = torch.ones(self.num_assets, dtype=torch.float64) + self.exp_returns[t,:]
        Sigma_hat_t = self.Sigma_r_tilde + torch.ger(one_plus_exp_ret, one_plus_exp_ret)

        # matrix A_t = diag(1 + r_bar_t)
        A_t = torch.diag(torch.ones(self.num_assets, dtype=torch.float64) + self.exp_returns[t,:])

        # update parameters of Q_function using next-step F-function values
        self.update_Q_params(t, A_t, Sigma_hat_t)

        # iterate between policy evaluation and policy improvement
        while self.iter_counts[t] < max_iter:

            curr_u_bar_prior = self.u_bar_prior[t,:].clone()
            curr_v_bar_prior = self.v_bar_prior[t,:,:].clone()

            # compute parameters of F-function for this step from parameters of Q-function
            self.update_F_params(t)

            # Policy iteration step: update parameters of the prior policy distribution
            # with given Q- and F-function parameters
            self.update_policy_params(t)

            # difference between the current value of u_bar_prior and the previous one
            err_u_bar = torch.sum((curr_u_bar_prior - self.u_bar_prior[t,:])**2)

            # divide by num_assets in err_v_bar to get both errors on a comparable scale
            err_v_bar = (1/self.num_assets)*torch.sum((curr_v_bar_prior - self.v_bar_prior[t,:,:])**2)

            # choose the difference from the previous iteration as the maximum of the two errors
            tol = torch.max(err_u_bar, err_v_bar)  # tol = 0.5*(err_u_bar + err_v_bar)

            self.iter_counts[t] += 1
            # Repeat the calculation of Q- and F-values
            if tol <= err_tol:
                break

    def update_Q_params(self, t, A_t, Sigma_hat_t):
        """
        Update the current (time-t) parameters of Q-function from (t+1)-parameters of F-function
        Parameters
        ----------
        t : the t-th step of model that the function is running
        A_t : diag(1+expected returns of assets)
        Sigma_hat_t : not used
        """

        ones = torch.ones(self.num_assets, dtype=torch.float64)
        one_plus_exp_ret = torch.ones(self.num_assets, dtype=torch.float64) + self.exp_returns[t,:]

        self.Q_xx[t,:,:] = (self.R_xx[t,:,:].clone()
                            + self.gamma *((A_t.clone().mm(self.F_xx[t+1,:,:].clone())).mm(A_t.clone())
                                           + self.Sigma_r_tilde.clone() * self.F_xx[t+1,:,:].clone()))


        self.Q_ux[t,:,:] = (self.R_ux[t,:,:].clone()
                            + 2 * self.gamma * ((A_t.clone().mm(self.F_xx[t+1,:,:].clone())).mm(A_t.clone())
                                                + self.Sigma_r_tilde.clone() * self.F_xx[t+1,:,:].clone())
                            )

        self.Q_uu[t,:,:] = (self.R_uu[t,:,:].clone()
                            + self.gamma * ((A_t.clone().mm(self.F_xx[t+1,:,:].clone())).mm(A_t.clone())
                                            + self.Sigma_r_tilde.clone() * self.F_xx[t+1,:,:].clone())
                            - self.Omega_mat.clone()
                            )


        self.Q_x[t,:] = self.R_x[t,:].clone() + self.gamma * A_t.clone().mv(self.F_x[t+1,:].clone())
        self.Q_u[t,:] = self.R_u[t,:].clone() + self.gamma * A_t.clone().mv(self.F_x[t+1,:].clone())
        self.Q_0[t] = self.R_0[t].clone() + self.gamma * self.F_0[t+1].clone()

    def update_F_params(self, t):
        """
        Update the current (time-t) parameters of F-function from t-parameters of G-function
        This is a policy evaluation step: it uses the current estimations of the mean parameters of the policy
        Parameters
        ----------
        t : the t-th step of model that the function is running
        """

        # produce auxiliary parameters U_t, W_t, Sigma_tilde_t
        U_t = (self.beta.clone() * self.Q_ux[t,:,:].clone()
               + self.Sigma_prior_inv[t,:,:].clone().mm(self.v_bar_prior[t,:,:].clone()))
        W_t = (self.beta.clone() * self.Q_u[t,:].clone()
               + self.Sigma_prior_inv[t,:,:].clone().mv(self.u_bar_prior[t,:]).clone())
        Sigma_p_bar = self.Sigma_prior_inv[t,:,:].clone() - 2 * self.beta.clone() * self.Q_uu[t,:,:].clone()
        Sigma_p_bar_inv = torch.inverse(Sigma_p_bar + self.reg_mat)

        # update parameters of F-function
        self.F_xx[t,:,:] = self.Q_xx[t,:,:].clone() + (1/(2*self.beta.clone()))*(U_t.t().mm(Sigma_p_bar_inv.clone().mm(U_t))
                                                                                 - self.v_bar_prior[t,:,:].clone().t().mm(
                    self.Sigma_prior_inv[t,:,:].clone().mm(self.v_bar_prior[t,:,:].clone())))


        self.F_x[t,:] = self.Q_x[t,:].clone() + (1/self.beta.clone())*(U_t.mv(Sigma_p_bar_inv.clone().mv(W_t))
                                                                       - self.v_bar_prior[t,:,:].clone().mv(
                    self.Sigma_prior_inv[t,:,:].clone().mv(self.u_bar_prior[t,:].clone())))


        self.F_0[t] = self.Q_0[t].clone() + ((1/(2*self.beta.clone()))*(W_t.dot(Sigma_p_bar_inv.clone().mv(W_t))
                                                                        - self.u_bar_prior[t,:].clone().dot(
                    self.Sigma_prior_inv[t,:,:].clone().mv(self.u_bar_prior[t,:].clone())))
                                             - (1/(2*self.beta.clone())) * (torch.log(torch.det(self.Sigma_prior[t,:,:].clone() +
                                                                                                self.reg_mat))
                                                                            - torch.log(torch.det(Sigma_p_bar_inv.clone() + self.reg_mat))))

    def update_policy_params(self, t):
        """
        Update parameters of the Gaussian policy using current parameters in the F-functions and G-functions
        Parameters
        ----------
        t : the t-th step of model that the function is running
        """

        new_Sigma_prior_inv = self.Sigma_prior_inv[t,:,:].clone() - 2 * self.beta.clone() * self.Q_uu[t,:,:].clone()

        Sigma_prior_new = torch.inverse(new_Sigma_prior_inv + self.reg_mat)

        # update parameters using the previous value of Sigma_prior_inv
        self.u_bar_prior[t,:] = Sigma_prior_new.mv(self.Sigma_prior_inv[t,:,:].clone().mv(self.u_bar_prior[t,:].clone())
                                                   + self.beta.clone() * self.Q_u[t,:].clone())


        self.v_bar_prior[t,:,:] = Sigma_prior_new.clone().mm(self.Sigma_prior_inv[t,:,:].clone().mm(self.v_bar_prior[t,:,:].clone())
                                                             + self.beta.clone() * self.Q_ux[t,:,:].clone())

        # and then assign the new inverse covariance for the prior for the next iteration
        self.Sigma_prior[t,:,:] = Sigma_prior_new.clone()
        self.Sigma_prior_inv[t,:,:] = new_Sigma_prior_inv.clone()

        # also assign the same values for the previous time step
        if t > 0:
            self.Sigma_prior[t-1,:,:] = self.Sigma_prior[t,:,:].clone()
            self.u_bar_prior[t-1,:] = self.u_bar_prior[t,:].clone()
            self.v_bar_prior[t-1,:,:] = self.v_bar_prior[t,:,:].clone()

    def trajs_to_torch_tensors(self, trajs):
        """
        Convert trajectories data from a list of lists into Torch tensors
        Parameters
        ----------
        trajs : trajectories
        """
        num_trajs = len(trajs)

        self.data_xvals = torch.zeros(num_trajs, self.num_steps, self.num_assets, dtype=torch.float64)
        self.data_uvals = torch.zeros(num_trajs, self.num_steps, self.num_assets, dtype=torch.float64)

        for n in range(num_trajs):
            for t in range(self.num_steps):
                self.data_xvals[n,t,:] = torch.tensor(trajs[n][t][0], dtype=torch.float64).clone()
                self.data_uvals[n,t,:] = torch.tensor(trajs[n][t][1], dtype=torch.float64).clone()

    def compute_reward_on_traj(self,
                               t,
                               x_t, u_t):
        """
        Given time t and corresponding values of vectors x_t, u_t, compute the accumulative reward at the t-th step
        Parameters
        ----------
        t : the t-th step of model that the function is running
        x_t : cash in assets at the t-th step
        u_t : cash change in assets at the t-th step

        Returns
        -------
        Computed reward (aux_xx + aux_ux + aux_uu + aux_x + aux_u + aux_0)
        """

        aux_xx = x_t.dot(self.R_xx[t,:,:].clone().mv(x_t))
        aux_ux = u_t.dot(self.R_ux[t,:,:].clone().mv(x_t))
        aux_uu = u_t.dot(self.R_uu[t,:,:].clone().mv(u_t))
        aux_x = x_t.dot(self.R_x[t,:].clone())
        aux_u = u_t.dot(self.R_u[t,:].clone())
        aux_0 = self.R_0[t].clone()

        return aux_xx + aux_ux + aux_uu + aux_x + aux_u + aux_0

    def compute_G_fun_on_traj(self,
                              t,
                              x_t, u_t):
        """
        Given time t and corresponding values of vectors x_t, u_t, compute the accumulative reward as of the t-th step
        Parameters
        ----------
        t : the t-th step of model that the function is running
        x_t : cash in assets at the t-th step
        u_t : cash change in assets at the t-th step

        Returns
        -------
        Computed accumulative reward (aux_xx + aux_ux + aux_uu + aux_x + aux_u + aux_0)
        """

        aux_xx = x_t.dot(self.Q_xx[t,:,:].clone().mv(x_t))
        aux_ux = u_t.dot(self.Q_ux[t,:,:].clone().mv(x_t))
        aux_uu = u_t.dot(self.Q_uu[t,:,:].clone().mv(u_t))
        aux_x = x_t.dot(self.Q_x[t,:].clone())
        aux_u = u_t.dot(self.Q_u[t,:].clone())
        aux_0 = self.Q_0[t].clone()

        return aux_xx + aux_ux + aux_uu + aux_x + aux_u + aux_0

    def compute_F_fun_on_traj(self,
                              t,
                              x_t):
        """
        Given time t and corresponding values of vectors x_t, u_t, compute the total reward at the t-th step
        Parameters
        ----------
        t : the t-th step of model that the function is running
        x_t : cash in assets at the t-th step

        Returns
        -------
        Computed F function value at the t-th step (aux_xx + aux_x + aux_0)
        """

        aux_xx = x_t.dot(self.F_xx[t,:,:].clone().mv(x_t))
        aux_x = x_t.dot(self.F_x[t,:].clone())
        aux_0 = self.F_0[t].clone()

        return aux_xx + aux_x + aux_0

    def MaxEntIRL(self,
                  trajs,
                  learning_rate,
                  err_tol, max_iter):
        """
        Estimate parameters of the reward function using MaxEnt IRL, given the input trajectory.
        Parameters
        ----------
        trajs : a list of trajectories. Each trajectory is a list of state-action pairs, stored as a tuple.
                We assume each trajectory has the same length
        learning_rate : learning rate
        err_tol : error tolerance
        max_iter : maximum iteration
        """

        # omega is a tunable parameter that determines the cost matrix self.Omega_mat
        omega_init = 15.0
        self.omega = torch.tensor(omega_init, requires_grad=True, dtype=torch.float64)

        beta_init = 50  # Beta is fixed and not a learned parameter.
        self.beta = torch.tensor(beta_init, requires_grad=True, dtype=torch.float64)

        reward_params = [self.lambd, self.eta, self.rho, self.omega, self.beta]

        print("Omega mat...")
        self.Omega_mat = self.omega * torch.eye(self.num_assets, dtype=torch.float64)
        print("g learning...")
        self.reset_prior_policy()
        self.G_learning(err_tol, max_iter)
        print("intialize optimizer...")
        optimizer = optim.Adam(reward_params, lr=learning_rate)
        print("zero grad...")
        optimizer.zero_grad()

        num_trajs = len(trajs)
        print("trajs_to_torch_tensors...")

        # fill in Torch tensors for the trajectory data
        self.trajs_to_torch_tensors(trajs)
        print("constructing zero tensors...")
        self.realized_rewards = torch.zeros(num_trajs, self.num_steps, dtype=torch.float64, requires_grad=True)
        self.realized_cum_rewards = torch.zeros(num_trajs, dtype=torch.float64, requires_grad=True)
        print("constructing zero tensors...")
        self.realized_G_fun = torch.zeros(num_trajs, self.num_steps, dtype=torch.float64, requires_grad=True)
        self.realized_F_fun = torch.zeros(num_trajs, self.num_steps, dtype=torch.float64, requires_grad=True)
        print("constructing zero tensors...")
        self.realized_G_fun_cum = torch.zeros(num_trajs, dtype=torch.float64, requires_grad=True)
        self.realized_F_fun_cum = torch.zeros(num_trajs, dtype=torch.float64, requires_grad=True)
        print("done...")

        num_iter_IRL = 3

        for i in range(num_iter_IRL):
            print('GIRL iteration = ', i)

            self.Omega_mat = self.omega * torch.eye(self.num_assets, dtype=torch.float64)

            for n in range(101):
                if n % 100 == 0:
                    print(n)
                for t in range(self.num_steps):

                    # compute rewards obtained at each step for each trajectory
                    # given the model parameters
                    self.realized_rewards[n,t] = self.compute_reward_on_traj(t,
                                                                             self.data_xvals[n,t,:],
                                                                             self.data_uvals[n,t,:])

                    # compute the log-likelihood by looping over trajectories
                    self.realized_G_fun[n,t] = self.compute_G_fun_on_traj(t,
                                                                          self.data_xvals[n,t,:],
                                                                          self.data_uvals[n,t,:])

                    self.realized_F_fun[n,t] = self.compute_F_fun_on_traj(t,
                                                                          self.data_xvals[n,t,:])

                self.realized_cum_rewards[n] = self.realized_rewards[n,:].sum().clone()
                self.realized_G_fun_cum[n] = self.realized_G_fun[n,:].sum().clone()
                self.realized_F_fun_cum[n] = self.realized_F_fun[n,:].sum().clone()

            # the negative log-likelihood will not include terms ~ Sigma_p as we do not optimize over its value
            loss = - self.beta.clone()*(self.realized_G_fun_cum.sum().clone() - self.realized_F_fun_cum.sum().clone())

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            print('Iteration = ', i)
            print('Loss = ', loss.detach().numpy())

        print('Done optimizing reward parameters')


class g_learn:
    def __init__(self, num_steps, num_risky_assets, x_vals_init):
        """
        A g-learning rolling class.
        This class will be rolling on series of time steps from G_learning_portfolio_opt class.
        After it calling G_learning_portfolio_opt class, it will update parameters into G_learning_portfolio_opt class
        and get to the next series of time steps.
        Parameters
        ----------
        num_steps : g-learning model number of steps (months here)
        num_risky_assets : number of assets
        x_vals_init : initial cash assigned to each asset at step 1
        """
        self.num_risky_assets = num_risky_assets
        self.num_steps = num_steps
        self.num_assets = self.num_risky_assets
        self.expected_risky_returns = np.empty((self.num_steps, self.num_risky_assets))
        self.sigma = np.empty((self.num_risky_assets, self.num_risky_assets))

        lambd = 0.001
        omega = 1.0
        beta = 1000.0
        gamma = 0.95
        eta = 1.5  # 1.3 # 1.5 # 1.2
        rho = 0.4

        self.reward_params = [lambd, omega, eta, rho]  # Parameters in reward function
        self.beta = beta
        self.gamma = gamma
        # Time step unit (day/week/month/year)
        dt = 1/12  # month
        self.x_vals_init = x_vals_init
        self.x_t = self.x_vals_init[:]
        # It will be arrays of cash value for each asset right after action
        self.x_next = 0
        # It will be arrays of cash value change (action) for each asset
        self.u_t = 0
        # benchmark portfolio growth rate
        target_return = 0.8
        self.benchmark_portf = [x_vals_init.sum() * np.exp(dt * target_return)]
        for i in range(1, self.num_steps):
            self.benchmark_portf.append(self.benchmark_portf[i-1]*np.exp(dt * target_return))

        # Record trajectories
        self.trajs = [[]]*self.num_steps  # Trajectory of one unit of model time steps
        self.trajs_all = []  # Trajectory of all time steps (time steps = several times of model time steps)

        # It will be g-learning class, however, after model time steps we need to define a new g-learning class
        self.learner = []

    def param_update(self, t, exp_returns, sigma):
        """
        Update class expected return matrix after and including the t+1th step,
        and update assets return covariance matrix
        Parameters
        ----------
        t : the step number (t+1) to be updated
        exp_returns : expected return at the t+1 step
        sigma : assets return covariance matrix at the t+1 step
        """
        self.expected_risky_returns[t:] = exp_returns
        self.sigma = sigma

    def run(self, t, exp_returns, sigma):
        """
        Execute g-learning by calling g-learning class with input expected return matrix and covariance matrix updated
        at the t+1th step, then record the action (cash injection) at the t+1th step and output assets weight
        Parameters
        ----------
        t : the step number (t+1)
        exp_returns :expected return at the t+1th step
        sigma :assets return covariance matrix at the t+1th step

        Returns
        -------
        optimal assets weight at the t+1th step
        """
        # update the g-learning rolling class expected return matrix and covariance matrix
        self.param_update(t, exp_returns, sigma)
        # define a g-learning class
        self.learner = G_learning_portfolio_opt(self.num_steps,
                                                self.reward_params,
                                                self.beta,
                                                self.benchmark_portf,
                                                self.gamma,
                                                self.num_risky_assets,
                                                self.expected_risky_returns,  # array of shape num_steps x num_stocks
                                                self.sigma,     # covariance matrix of returns of risky matrix
                                                self.x_vals_init,  # array of initial values of len (num_stocks+1)
                                                use_for_WM=True)  # use for wealth management tasks
        # initialize time-dependent parameters of prior policy for the g-learning class
        self.learner.reset_prior_policy()
        error_tol = 1.e-8
        max_iter_RL = 200
        # train the g-learning model
        self.learner.G_learning(error_tol, max_iter_RL)

        # generate actions (cash injection) based on trained optimal policy
        mu_t = self.learner.u_bar_prior[t,:] + self.learner.v_bar_prior[t,:].mv(torch.tensor(self.x_t))
        self.u_t = np.random.multivariate_normal(mu_t.detach().numpy(), self.learner.Sigma_prior[t,:].detach().numpy())

        # compute new values of x_t (cash for each asset)
        self.x_next = self.x_t + self.u_t
        # compute optimal weight given new values of x_t
        weight = self.x_next / self.x_next.sum()

        return weight

    def update(self, t, returns):
        """
        Record models trajectory and update assets cash with actual returns
        Parameters
        ----------
        t : the step number (t+1)
        returns : actual returns at the t+1th step
        """
        self.x_next = (1+returns)*self.x_next

        self.trajs[t:] = [[]]*(self.num_steps - t)
        self.trajs[t] = [self.x_t, self.u_t]
        self.trajs_all.append((self.x_t, self.u_t))

        # rename
        self.x_t = self.x_next

    def update_before_step_1(self, t):
        """
        At the step 1, we need to update the initial cash assigned to each asset
        and redefine benchmark portfolio based on the new initial assets cash
        Parameters
        ----------
        t : the step number (t+1)
        """
        if t == 0:
            self.x_vals_init = self.x_t

            dt = 1/12
            target_return = 0.8
            self.benchmark_portf = [self.x_vals_init.sum() * np.exp(dt * target_return)]
            for i in range(1, self.num_steps):
                self.benchmark_portf.append(self.benchmark_portf[i-1]*np.exp(dt * target_return))


def g_learn_rolling(t, g_learner, exp_returns, sigma, returns):
    """
    Call g-learning rolling class functions to output optimal weight and update g-learning rolling class
    Parameters
    ----------
    t : the step number (t+1)
    g_learner : the g-learning rolling class
    exp_returns : expected return at the t+1th step
    sigma : assets return covariance matrix at the t+1th step
    returns : actual returns at the t+1th step

    Returns
    -------
    optimal weight at the t+1th step, and the updated g_learning rolling class
    """
    if t == 0:
        g_learner.update_before_step_1(t)

    weight = g_learner.run(t, exp_returns, sigma)
    g_learner.update(t, returns)

    return weight, g_learner




