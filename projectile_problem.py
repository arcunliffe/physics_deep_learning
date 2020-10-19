"""

Deep learning to solve the physics projectile problem: Given an initial velocity
and angle of a projectile, predict the distance traveled, max height, and
travel time.

Alex Cunliffe
2020-10-17

"""

import itertools

import numpy as np
from torch import nn, optim

from models import MLPRegressor
from datasets import projectile_dataset
from training_utils import *


def hyperparameter_combinations(hyperparameters):
    """
    Generate all possible combinations of hyperparameters.
    Inputs:
       hyperparameters: dict containing pairs of hyperparameter names and
          set of values to try. Example:
          {"learning_rate": [0.1, 0.2],
           "epochs": [100, 1000]}
    Outputs:
       list<dict> of each hyperparameter set. Example:
       [{"learning_rate": 0.1, "epochs": 100},
        {"learning_rate": 0.1, "epochs": 1000},
        {"learning_rate": 0.2, "epochs": 100},
        {"learning_rate": 0.2, "epochs": 1000}]
    """
    value_combinations = list(itertools.product(*hyperparameters.values()))
    hyperparam_combos = [{k: v for k, v in zip(hyperparameters.keys(), vals)}
                         for vals in value_combinations]
    return hyperparam_combos


def train_mlpregressor(model_hyperparams, loss_function, data, display_loss):
    """hHlper function to train the MLPRegressor model with a set of model
       hyperparameters using Adam optimization."""
    model = MLPRegressor(model_hyperparams["layer_sizes"])
    optimizer = optim.Adam(model.parameters(), lr=model_hyperparams["learning_rate"])
    trained_model = train_network(
        model, data, model_hyperparams["epochs"], loss_function, optimizer,
        display_loss=display_loss
    )
    return trained_model


if __name__ == "__main__":

    # build projectile dataset
    x, y, normalization_params = projectile_dataset(5000, normalize=True)

    # create a hold-out test set
    train_fraction = 0.9
    x_trainval, y_trainval, x_test, y_test = train_test_split(x, y, train_fraction)

    # create train and validation sets
    train_fraction = 0.8
    x_train, y_train, x_val, y_val = train_test_split(x_trainval, y_trainval, train_fraction)
    data = {"x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val}

    # generate combinations of hyperparameters to tune
    hyperparameters = {
        "layer_sizes": [[x.shape[1], 64, y.shape[1]],
                        [x.shape[1], 64, 128, 64, y.shape[1]],
                        [x.shape[1], 64, 128, 256, 128, 64, y.shape[1]]],
                        #[x.shape[1], 64, 128, 256, 512, 256, 128, 64, y.shape[1]]],
        "learning_rate": [0.1, 0.05, 0.01, 0.005, 0.001],
        "epochs": [100]
    }
    hyperparameter_combos = hyperparameter_combinations(hyperparameters)

    # hyperparameter tuning
    loss_function = nn.MSELoss()
    val_losses = []
    for hyperparams in hyperparameter_combos:
        print("Training model with hyperparameters:", hyperparams)
        trained_model = train_mlpregressor(hyperparams, loss_function, data, display_loss=False)

        # compute validation set loss
        val_loss = loss_function(trained_model(data["x_val"]), data["y_val"])
        print("Validation set loss:", round(val_loss.item(), 3))
        print()
        val_losses.append(val_loss.item())

    # identify hyperparameter set that minimized validation loss
    optimal_hyperparams = hyperparameter_combos[np.argmin(val_losses)]
    print("Optimal model hyperparameters:", optimal_hyperparams)

    # train model using all training/validation data, display train/test set loss
    data = {"x_train": x_trainval,
            "y_train": y_trainval,
            "x_val": x_test,
            "y_val": y_test}
    trained_model = train_mlpregressor(optimal_hyperparams, loss_function, data, display_loss=True)
