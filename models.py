"""

torch model architectures for deep learning

Alex Cunliffe
2020-10-18

"""

from torch import nn

class MLPRegressor(nn.Module):
    """
    A simple multilayer perceptron to predict a continuous output (i.e., a
    regression problem).
    """
    def __init__(self, layer_sizes):
        """
        Set up the layers in the MLP
        Inputs:
           layer_sizes: list<int>, number of nodes in each layer
        """
        super(MLPRegressor, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x):
        """
        Forward pass through the MLP. Applies each layer followed by a ReLU
        activation function.
        Inputs:
           x: torch.Tensor, input feature data
        """
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                # final layer
                return layer(x)
            else:
                x = layer(x)
                x = nn.ReLU()(x)
