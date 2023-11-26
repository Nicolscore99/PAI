"""Solution."""
import numpy as np
import scipy
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, DotProduct, RBF, ExpSineSquared, WhiteKernel

import matplotlib.pyplot as plt
from matplotlib import gridspec

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 10  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.

        # Data buckets
        self.x_sample = np.array([]).reshape(-1, DOMAIN.shape[0])
        self.f_sample = np.array([]).reshape(-1, DOMAIN.shape[0])
        self.v_sample = np.array([]).reshape(-1, DOMAIN.shape[0])
        # self.logv_sample = np.array([]).reshape(-1, DOMAIN.shape[0])
        # self.gv_sample = np.array([]).reshape(-1, DOMAIN.shape[0])

        # Define the priors
        # Kernel for f mapping (Matern kernel)
        self.f_nu = 2.5
        self.f_length_scale = 10
        # self.f_kernel = Matern(length_scale=self.f_length_scale, nu=self.f_nu)
        self.f_kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.0225, noise_level_bounds='fixed')

        self.v_nu = 2.5
        self.v_length_scale = 1.0
        self.v_var = math.sqrt(2)
        self.v_prior_mean = 4.0
        # self.v_kernel = DotProduct() + Matern(length_scale=self.v_length_scale, nu=self.v_nu)
        # self.v_kernel = ConstantKernel(constant_value=self.v_prior_mean)*RBF(length_scale=self.v_length_scale)
        # self.v_kernel = ExpSineSquared(length_scale=self.v_length_scale, periodicity=1.0, periodicity_bounds=(1e-2, 1e1))
        self.v_kernel = Matern(nu=2.5) + DotProduct() + WhiteKernel(noise_level=0.00001, noise_level_bounds='fixed')

        # Define the GP models
        self.f_gp = GaussianProcessRegressor(kernel=self.f_kernel)
        self.v_gp = GaussianProcessRegressor(kernel=self.v_kernel)

        # Constraint violation penalty
        self.lambda_ = 0.01
        # Eploration-exploitation trade-off parameter
        self.f_beta = 1.0
        self.v_beta = 0.5 / self.lambda_
  


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        # Implement some custom rule  to include randomness

        # If the last two points were unsafe, then sample randomly 
        # if self.x_sample.size > 1 and self.v_sample[-1] > SAFETY_THRESHOLD and self.v_sample[-2] > SAFETY_THRESHOLD:
        #     print("You are in a bad place, let's get out of here...")
        #     # sample randomly
        #     x = np.random.uniform(*DOMAIN[0])
        #     return x

        # If we have a large portion of unsafe points, then create more incentive to explore
        # if self.x_sample.size > 3 and np.mean(self.v_sample > SAFETY_THRESHOLD) > 0.5:
        #     # This way we should choose points where the uncertainty on v is high and therefore do much more exploration
        #     self.v_beta = 2.0*self.v_beta
        #     print("We are in a bad place, let's explore more...")


        # If we have a large portion of safe points, then create more incentive to exploit

        # opts = []
        # results = []
        # for i in range(5):
        #     opt = self.optimize_acquisition_function()
        #     opts.append(opt)
        #     opt = np.atleast_2d(opt)
        #     mean, std = self.f_gp.predict(opt, return_std=True)
        #     results.append(mean)

        # Return opts value with the highest mean

        # return self.optimize_acquisition_function()

        return np.atleast_2d(self.optimize_acquisition_function())

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            # print(x0.shape)
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.

        # compute the mean and std of the GP model
        f_mean, f_std = self.f_gp.predict(x, return_std=True)
        v_mean, v_std = self.v_gp.predict(x, return_std=True)

        f_mean += 4.0

        # compute the acquisition function
        # Here I've changed the plusses and minuses to account for the fact that

        af_value = f_mean - self.f_beta * f_std - self.lambda_*max((v_mean + self.v_beta * v_std),4)

        return af_value

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.

        # stack the newly obtained data point onto the existing data points
        self.x_sample    = np.vstack((self.x_sample, x))
        self.f_sample    = np.vstack((self.f_sample, f))
        self.v_sample    = np.vstack((self.v_sample, v))
        # self.logv_sample = np.vstack((self.logv_sample, math.log(v)))
        # self.gv_sample   = np.vstack((self.gv_sample, v - self.v_min))

        # update the GP model
        self.f_gp.fit(self.x_sample, self.f_sample)
        self.v_gp.fit(self.x_sample, self.v_sample)


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.

        # # plot the GP model
        # if self.x_sample.size > 1:
        #     self.plot(plot_recommendation=True)

        valid_samples = self.f_sample[self.v_sample < SAFETY_THRESHOLD]
        solution = valid_samples[np.argmax(valid_samples)]

        return solution

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """

        print("Plotting...")

        fig, ax = plt.subplots(figsize=(20, 10))

        green = (0, 172, 0, 1)
        light_green = (0, 172, 0, 0.3)
        red = (172, 0, 0, 1)
        light_red = (172, 0, 0, 0.3)

        # Plot objective function
        x = np.linspace(*DOMAIN[0], 100)[:, None]
        f_x = np.vectorize(f)(x)
        ax.plot(x, f_x, 'k', lw=1, zorder=9, label="Objective function $f(x)$")
        # plot the constraint function
        v_x = np.vectorize(v)(x)
        ax.plot(x, v_x, '--k', lw=1, zorder=9, label="Constraint function $v(x)$")
        # plot the constraint threshold
        ax.plot(x, np.ones(x.shape)*SAFETY_THRESHOLD, '--k', lw=2, zorder=9, label="Safety threshold")

        # Plot GP posterior of all points we have sampled
        self.f_gp.fit(self.x_sample, self.f_sample)
        f_mean, f_std = self.f_gp.predict(x, return_std=True)
        print("x", x)
        print("f_samle", self.f_sample)
        print("f_mean", f_mean)
        ax.plot(x, f_mean, 'g', lw=1, zorder=9, label="Posterior mean $f(x)$")
        ax.fill_between(x[:,0], f_mean - f_std, f_mean + f_std,
                        color=[light_green]*100, alpha=0.2,
                        label="Confidence region $\pm$ 1 std. dev.")
        ax.plot(self.x_sample, self.f_sample, 'gx', mew=1, label="Observed data $f(x)$")

        # Plot GP posterior of all points we have sampled
        v_mean, v_std = self.v_gp.predict(x, return_std=True)
        print(v_mean)
        ax.plot(x, v_mean, 'r', lw=1, zorder=9, label="Posterior mean $v(x)$")
        ax.fill_between(x[:, 0], v_mean - v_std, v_mean + v_std,
                        color=[light_red]*100, alpha=0.2,
                        label="Confidence region $\pm$ 1 std. dev.")
        ax.plot(self.x_sample, self.v_sample, 'rx', mew=1, label="Observed data $v(x)$")

        # define axis bounds
        ax.set_xlim(*DOMAIN[0])
        ax.set_ylim(-10, 20)

        plt.legend()

        plt.show()

# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    # mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    # return - np.linalg.norm(x - mid_point, 2)

    return (-x + 0.3*(x-2)*(x-8) - 1/2**(x-4)) + 10


def v(x: float):
    """Dummy SA"""
    # return 2.0
    return 5 + np.sin(1.1*x - 2.0)*5.0 + x / 4.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)

    agent.plot()

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
