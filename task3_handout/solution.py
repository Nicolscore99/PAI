"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, DotProduct

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


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
        self.f_length_scale = 0.5
        self.f_kernel = Matern(length_scale=self.f_length_scale, nu=self.f_nu)

        self.v_nu = 2.5
        self.v_length_scale = 0.5
        self.v_kernel = DotProduct() + Matern(length_scale=self.v_length_scale, nu=self.v_nu)

        # Define the GP models
        self.f_gp = GaussianProcessRegressor(kernel=self.f_kernel)
        self.v_gp = GaussianProcessRegressor(kernel=self.v_kernel)

        # Eploration-exploitation trade-off parameter
        self.f_beta = 2.0
        self.v_beta = 2.0
        # Constraint violation penalty
        self.lambda_ = 0.5


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

        return self.optimize_acquisition_function()

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

        # compute the acquisition function
        # TODO: Maybe 0 needs to be replaced by the minimum of the constraint function
        af_value = f_mean + self.f_beta * f_std + self.lambda_*max((v_mean + self.v_beta * v_std), 0)

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
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


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
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

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
