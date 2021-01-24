import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDQN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, state_size*6)
        self.fc2 = nn.Linear(state_size*6, state_size*4)
        self.fc3 = nn.Linear(state_size*4, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        hidden1 = F.relu(self.fc1(state))
        hidden2 = F.relu(self.fc2(hidden1))
        action_values = self.fc3(hidden2)
        
        return action_values 