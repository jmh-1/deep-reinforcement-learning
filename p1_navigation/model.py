import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.seq = nn.Sequential(
            nn.Linear(state_size, 50),
#             nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 50),
#             nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Linear(50, action_size),
        )
        "*** YOUR CODE HERE ***"

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.seq(state)