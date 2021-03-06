"""

Tests for datasets.py

Alex Cunliffe
2020-10-18

"""

import math

from datasets import *

EPS = 0.001

def test_projectile_dataset():
    """tests that projectile_dataset yields expected results and agrees with
       physics"""
    x, y, _ = projectile_dataset(2, normalize=False)
    assert x.shape[1] == 2
    assert y.shape[1] == 3

    velocity, angle_radians = np.array(x)[1]
    x_displacement, max_height, time = np.array(y)[1]

    x_velocity = math.cos(angle_radians) * velocity
    y_velocity = math.sin(angle_radians) * velocity

    # compute time
    t = 2 * y_velocity / 9.8
    assert abs(t - time) < EPS

    # compute distance
    distance = x_velocity * t
    assert abs(distance - x_displacement) < EPS

    # compute height
    height = 0.5 * (9.8) * (0.5 * t)**2
    assert abs(height - max_height) < EPS

def test_random_dataset():
    """test that the random dataset has the expected shape"""
    n_samples = 100
    input_size = 2
    output_size = 3
    x, y = random_dataset(n_samples, input_size, output_size)
    assert x.shape[0] == n_samples
    assert y.shape[0] == n_samples
    assert x.shape[1] == input_size
    assert y.shape[1] == output_size
