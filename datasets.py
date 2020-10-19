"""

Generate datasets of inputs/outputs for model training.

Alex Cunliffe
2020-10-18

"""

import random

import numpy as np
import torch


def projectile_dataset(n_samples, normalize=True):
    """
    Generate dataset for the projectile problem - uses actual physics to compute,
    based on a projectile's initial velocity and angle, the horizontal
    displacement, max height, and travel time of the projectile.
    Inputs:
       n_samples: int, number of samples to generate
       normalize: bool, if True, normalize the inputs and outputs by subtracting
          the mean and dividing by the standard deviation
    Outputs:
       x: torch.Tensor, velocity & angle for a set of projectiles
       y: torch.Tensor, associated horizontal displacement, max height, and
          travel time of projectiles
       normalization_params: dict; Empty if normalize is False. Otherwise contains:
          {"x_mean": np.array,
           "x_std": np.array,
           "y_mean": np.array,
           "y_std": np.array
    """
    initial_velocity = np.array(range(n_samples))
    angle = np.array(random.choices(range(90), k=n_samples))
    angle_radians = angle * np.pi / 180.0

    horizontal_velocity = initial_velocity * np.cos(angle_radians)
    vertical_velocity = initial_velocity * np.sin(angle_radians)
    time = 2 * vertical_velocity / 9.8
    horizontal_displacement = horizontal_velocity * time
    max_height = vertical_velocity * 0.5 * time + 0.5 * (-9.8) * (0.5 * time) ** 2

    x = np.vstack((initial_velocity, angle_radians)).T
    y = np.vstack((horizontal_displacement, max_height, time)).T

    if normalize:
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x_normalized = (x - x_mean) / x_std
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        y_normalized = (y - y_mean) / y_std

        normalization_params = {"x_mean": x_mean,
                                "x_std": x_std,
                                "y_mean": y_mean,
                                "y_std": y_std}

        return torch.from_numpy(x_normalized).float(), \
            torch.from_numpy(y_normalized).float(), \
            normalization_params

    return torch.from_numpy(x).float(), torch.from_numpy(y).float(), {}


def random_dataset(n_samples, n_input_features, n_output_features):
    """random dataset used for testing"""
    return torch.randn(n_samples, n_input_features), torch.randn(n_samples, n_output_features)
