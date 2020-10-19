"""

tests for training_utils.py

Alex Cunliffe
2020-10-18

"""

from datasets import random_dataset
from training_utils import *

def test_train_test_split():
    """test the train_fraction input to train_test_split"""

    # generate a dummy dataset
    n_samples = 100
    input_size = 2
    output_size = 3
    x, y = random_dataset(n_samples, input_size, output_size)

    train_fraction = 0.9
    x_train, y_train, x_test, y_test = train_test_split(x, y, train_fraction)
    assert x_train.shape[0] == int(train_fraction * n_samples)
    assert x_train.shape[1] == input_size
    assert y_train.shape[0] == int(train_fraction * n_samples)
    assert y_train.shape[1] == output_size
    assert x_train.shape[0] + x_test.shape[0] == n_samples
    assert y_train.shape[0] + y_test.shape[0] == n_samples
