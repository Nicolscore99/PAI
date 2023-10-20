import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.kernel_approximation import (RBFSampler, Nystroem)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn import pipeline

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        self.random_state = 42

        # TODO: Add custom initialization for your model here if necessary

        # self.kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))
        # self.kernel = Matern(length_scale=0.01, nu=2.5) + RationalQuadratic(length_scale=0.05, alpha=0.5) +  WhiteKernel(noise_level=1e-5)
        self.kernel = Matern(length_scale=0.01, nu=2.5) +  WhiteKernel(noise_level=1e-5)

        # self.feature_map = RBFSampler(gamma=1, n_components=2, random_state=self.random_state)
        self.feature_map = Nystroem(gamma=1, n_components=2, random_state=1)

        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.01, n_restarts_optimizer=10, random_state=42)

    def make_predictions(self, test_x_2D: np.ndarray, test_x_AREA: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_x_2D: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_x_AREA: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # TODO: Use your GP to estimate the posterior mean and stddev for each city_area here
        gp_mean = np.zeros(test_x_2D.shape[0], dtype=float)
        gp_std = np.zeros(test_x_2D.shape[0], dtype=float)

        # TODO: Use the GP posterior to form your predictions here
        gp_mean, gp_std = self.gp.predict(test_x_2D, return_std=True)

        predictions = gp_mean + np.ones(test_x_2D.shape[0], dtype=float) * self.y_mean

        return predictions, gp_mean, gp_std

    def fitting_model(self, train_y: np.ndarray,train_x_2D: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x_2D: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        # TODO: Fit your model here
        scaler = StandardScaler().fit(train_x_2D)
        self.y_mean = train_y.mean()

        gpr_pipeline = pipeline.Pipeline([
                                        ("scaler", scaler),
                                        ("feature_map", self.feature_map),
                                        ("gp", self.gp)
                                        ])

        gpr_pipeline.fit(train_x_2D, train_y - self.y_mean)

        pass

# You don't have to change this function
def cost_function(ground_truth: np.ndarray, predictions: np.ndarray, AREA_idxs: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param AREA_idxs: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(AREA_idx) for AREA_idx in AREA_idxs]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def is_in_circle(coor, circle_coor):
    """
    Checks if a coordinate is inside a circle.
    :param coor: 2D coordinate
    :param circle_coor: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coor[0] - circle_coor[0])**2 + (coor[1] - circle_coor[1])**2 < circle_coor[2]**2

# You don't have to change this function 
def determine_city_area_idx(visualization_xs_2D):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param visualization_xs_2D: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])
    
    visualization_xs_AREA = np.zeros((visualization_xs_2D.shape[0],))

    for i,coor in enumerate(visualization_xs_2D):
        visualization_xs_AREA[i] = any([is_in_circle(coor, circ) for circ in circles])

    return visualization_xs_AREA

# You don't have to change this function
def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs_2D = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    visualization_xs_AREA = determine_city_area_idx(visualization_xs_2D)
    
    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs_2D, visualization_xs_AREA)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax = ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def extract_city_area_information(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_x_2D = np.zeros((train_x.shape[0], 2), dtype=float)
    train_x_AREA = np.zeros((train_x.shape[0],), dtype=bool)
    test_x_2D = np.zeros((test_x.shape[0], 2), dtype=float)
    test_x_AREA = np.zeros((test_x.shape[0],), dtype=bool)

    #TODO: Extract the city_area information from the training and test features
    train_x_2D = train_x[:,0:2]
    train_x_AREA = train_x[:,2]
    test_x_2D = test_x[:,0:2]
    test_x_AREA = test_x[:,2]

    assert train_x_2D.shape[0] == train_x_AREA.shape[0] and test_x_2D.shape[0] == test_x_AREA.shape[0]
    assert train_x_2D.shape[1] == 2 and test_x_2D.shape[1] == 2
    assert train_x_AREA.ndim == 1 and test_x_AREA.ndim == 1

    return train_x_2D, train_x_AREA, test_x_2D, test_x_AREA

# takes a subsamlple of train_x, train_y and train_x_AREA of size trainset_size
def reduce_trainset_size(train_x: np.ndarray, train_y: np.ndarray, train_x_AREA: np.ndarray, trainset_size: int) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduces the size of the training set to the given size.
    :param train_x: Training features
    :param train_y: Training pollution concentrations
    :param train_x_AREA: Training city_area information
    :param trainset_size: Size of the reduced training set
    :return: Tuple of (reduced training features, reduced training pollution concentrations,
        reduced training city_area information)
    """
    assert train_x.shape[0] == train_y.shape[0] and train_x.shape[0] == train_x_AREA.shape[0]
    assert train_x.shape[1] == 2 and train_x_AREA.ndim == 1

    num_rows = train_x.shape[0]

    # Generate a random set of row indices to select
    selected_indices = np.random.choice(num_rows, trainset_size, replace=False)
    
    # Use the selected indices to slice the arrays
    reduced_train_x = train_x[selected_indices, :]
    reduced_train_y = train_y[selected_indices]
    reduced_train_x_AREA = train_x_AREA[selected_indices]

    return reduced_train_x, reduced_train_y, reduced_train_x_AREA

def return_mixed_reduced_trainset(self, train_x_incity, train_x_outofcity, train_y_incity, train_y_outofcity, total_size, portion):

    num_in_city = int(total_size * portion)
    num_outof_city = total_size - num_in_city

    selected_in_city_indices = np.random.choice(train_x_incity.shape[0], num_in_city, replace=False)
    selected_outof_city_indices = np.random.choice(train_x_outofcity.shape[0], num_outof_city, replace=False)

    reduced_train_x = np.concatenate((train_x_incity[selected_in_city_indices,:], train_x_outofcity[selected_outof_city_indices,:]), axis=0)
    reduced_train_y = np.concatenate((train_y_incity[selected_in_city_indices], train_y_outofcity[selected_outof_city_indices]), axis=0)

    return reduced_train_x, reduced_train_y

# you don't have to change this function
def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('task1_handout_d3d63876/train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('task1_handout_d3d63876/train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('task1_handout_d3d63876/test_x.csv', delimiter=',', skiprows=1)

    in_city_mask = (train_x[:,2] == 1)
    outof_city_mask = (train_x[:,2] == 0)

    train_x_in_city = train_x[np.where(in_city_mask)]
    train_x_outof_city = train_x[np.where(outof_city_mask)]
    train_y_in_city = train_y[np.where(in_city_mask)]
    train_y_outof_city = train_y[np.where(outof_city_mask)]
    
    reduced_train_x, reduced_train_y = return_mixed_reduced_trainset(train_x_in_city, train_x_outof_city, train_y_in_city, train_y_outof_city, total_size=1000, portion=0.8)

    # Extract the city_area information
    # train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(train_x, test_x)

    reduced_train_x_2D, reduced_train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(reduced_train_x, test_x)
    
    # print(train_x_2D.shape)
    # print(train_y.shape)

    # Reduce the size of the training set
    # train_x_2D, train_y, train_x_AREA = reduce_trainset_size(reduced_train_x_2D, train_y, train_x_AREA, trainset_size=1000)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(reduced_train_y,reduced_train_x_2D)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_x_2D, test_x_AREA)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
