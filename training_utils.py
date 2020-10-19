"""

Utility functions for training deep learning models using pytorch.

Alex Cunliffe
2020-10-18

"""

import random


def train_network(model, data, epochs, loss_function, optimizer, display_loss=True):
    """
    Train a pytorch model.
    Inputs:
       model: torch model, e.g., MLPRegressor()
       data: dict, contains at a minimum:
          {"x_train": <torch.Tensor>,
           "y_train": <torch.Tensor>,
           "x_val": <torch.Tensor>,
           "y_val": <torch.Tensor>}
        epochs: int, number of training epochs
        loss_function: torch loss function, e.g., torch.nn.MSELoss()
        optimizer: torch optimizer, e.g., torch.optim.Adam()
        display_loss: bool; If True, display the training/validation loss at
           each epoch. Defaults to True.
    Outputs:
       model: trained torch model
    """

    for _ in range(epochs):
        y_pred = model.forward(data["x_train"])
        loss = loss_function(y_pred, data["y_train"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute validation loss
        y_pred_val = model(data["x_val"])
        val_loss = loss_function(y_pred_val, data["y_val"])
        if display_loss:
            print("training/validation loss",
                  round(loss.item(), 3),
                  "/",
                  round(val_loss.item(), 3))

    return model


def train_test_split(x, y, train_fraction):
    """
       Randomly split data into training & testing set.
       Inputs:
          x: np.array or torch.Tensor, input feature vector
          y: np.array or torch.Tensor, output vector to predict
          train_fraction: float, number between 0 and 1 that specifies the
             proportion of the data that should be included in the training set
       Outputs:
          x_train: np.array or torch.Tensor, input feature vector for training set
          y_train: np.array or torch.Tensor, output vector to predict for training set
          x_test: np.array or torch.Tensor, input feature vector for test set
          y_test: np.array or torch.Tensor, output vector to predict for test set
    """

    n_samples = x.shape[0]
    indices = list(range(n_samples))
    random.shuffle(indices)

    n_training_samples = int(train_fraction * n_samples)
    train_indices = indices[:n_training_samples]
    test_indices = indices[n_training_samples:]

    x_train = x[train_indices, :]
    y_train = y[train_indices, :]
    x_test = x[test_indices, :]
    y_test = y[test_indices, :]

    return x_train, y_train, x_test, y_test
