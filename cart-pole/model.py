import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDQN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, state_size*6)
        self.fc2 = nn.Linear(state_size*6, state_size*6)
        self.fc3 = nn.Linear(state_size*6, action_size)
        
    def forward(self, state):
        hidden1 = F.relu(self.fc1(state))
        hidden2 = F.relu(self.fc2(hidden1))
        action_values = self.fc3(hidden2)
#         action_values = F.relu(self.fc1(state))
        
        return action_values
    
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, state_size*6)
        self.fc2 = nn.Linear(state_size*6, state_size*4)

        self.fc_state_1 = nn.Linear(state_size*4, 1)

        self.fc_advantages_1 = nn.Linear(state_size*4, action_size)
        
    def forward(self, state):
        # Common Part
        hidden1 = F.relu(self.fc1(state))
        hidden2 = F.relu(self.fc2(hidden1))
        
        # State Value Branch
        state_value = F.relu(self.fc_state_1(hidden2))

        # Advantage Values Branch
        advantage_values = F.relu(self.fc_advantages_1(hidden2))
        
        # Recombine branches with equation (9) from the paper (https://arxiv.org/pdf/1511.06581.pdf)
        action_values = state_value + (advantage_values - advantage_values.mean(dim=1).view(-1,1))
        
        return action_values
