import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, conv1_units=32, conv2_units=16, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # (B, 1, state_size) -> (B, conv1_units, state_size)
        self.conv1 = nn.Conv1d(1, conv1_units, kernel_size=3, padding=0)
        # (B, conv1_units, state_size) -> (B, conv1_units, state_size / 2)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(conv1_units)

        # (B, conv1_units, 1 + state_size / 2) -> (B, conv2_units, 1 + state_size / 2)
        self.conv2 = nn.Conv1d(conv1_units, conv2_units, kernel_size=3, padding=1)
        # (B, conv2_units, 1 + state_size / 2) -> (B, conv2_units, 1 + state_size / 4)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(conv2_units)

        self.post_conv_shape = conv2_units * int(state_size / 4)
        self.fc1 = nn.Linear(self.post_conv_shape, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = state.unsqueeze(1)
        x = self.bn1(self.pool1(self.conv1(state)))
        x = F.relu(x)

        x = self.bn2(self.pool2(self.conv2(x)))
        x = F.relu(x)

        x = x.reshape((state.shape[0], self.post_conv_shape))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
