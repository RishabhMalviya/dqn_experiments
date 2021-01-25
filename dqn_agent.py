import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, BUFFER_SIZE, BATCH_SIZE, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            self.BUFFER_SIZE (int): maximum size of buffer
            self.BATCH_SIZE (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=BUFFER_SIZE)  
        self.BATCH_SIZE = BATCH_SIZE
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.BATCH_SIZE)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class HyperparameterConfig:
    def __init__(self):
        self.EPS_START = 1.0
        self.EPS_END = 0.01
        self.EPS_DECAY = 0.995
        
        self.BUFFER_SIZE = int(1e5)  # replay buffer size
        self.GAMMA = 0.99            # discount factor

        self.BATCH_SIZE = 64         # minibatch size
        self.LR = 5e-4               # learning rate 
        self.UPDATE_EVERY = 4        # how often to the current DQN should learn
        
        self.HARD_UPDATE = False     # to hard update or not (with double DQN, one should)
        self.DOUBLE_DQN = False      # to use double DQN or not
        self.TAU = 1e-3              # for soft update of target parameters
        self.OVERWRITE_EVERY = 128   # how often to clone the local DQN to the target DQN
        
class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, QNetwork, hyperparameters):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            QNetwork (nn.Module): the class that defines the DQN architecture
            hyperparameters (HyperparameterConfig): hyperparameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Hyperparameters
        self.hp = hyperparameters

        # Q-Networks
        self.current_dqn = QNetwork(state_size, action_size, seed).to(device)        
        self.target_dqn = QNetwork(state_size, action_size, seed).to(device)
        
        # Optimizer for current DQN learning
        self.optimizer = optim.Adam(params=self.current_dqn.parameters(), lr=self.hp.LR)

        # Replay memory
        self.memory = ReplayBuffer(self.hp.BUFFER_SIZE, self.hp.BATCH_SIZE, seed)
        
        # Initialize the step count (to keep track of when to learn (current DQN) and update (target DQN))
        self.t_step = 0
    
    def _current_dqn_forward_pass(self, states):
        """Get the action values for the given state by performing a forward pass 
        through the current DQN with gradient calculations turend off.

        Args:
            states (torch.Tensor): May be a single state vector, or a minibatch
            
        Returns:
            torch.Tensor: the action-values, according to the current DQN
        """
        self.current_dqn.eval()
        with torch.no_grad():
            action_values = self.current_dqn(states)
        self.current_dqn.train()
        
        return action_values
    
    def act(self, state, eps=0.):
        """Determine the action based on the given state. This action choice is then sent to the environment, 
        which returns the next_state, reward, and whether or not the episode is done.

        Args:
            state (array-like): the current state
            eps (float, optional): The epsilon for the epsilon-greedy policy. Defaults to 0.

        Returns:
            int: The index of the chosen action
        """
        # Get the action values for the given state from the current DQN 
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_values = self._current_dqn_forward_pass(state)
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def step(self, state, action, reward, next_state, done):
        """Save the most recently acquired experience tuple into the replay memory.
        Every UPDATE_EVERY steps, this function also has the current DQN learn from a minibatch of experience tuples.

        Args:
            state (torch.Tensor): The initial state
            action (int): The action chosen by the agent from the initial state
            reward (float): The reward recieved from the action
            next_state (torch.Tensor): The next state after the action was executed 
            done (boolean): Whether or not the action causued the episode to terminate
        """
        # Save experience in the replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every self.UPDATE_EVERY time steps.
        self.t_step += 1
        if self.t_step % self.hp.UPDATE_EVERY == 0 and len(self.memory) > self.hp.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)
        
    def _hard_update(self):
        """Overwrite the target DQN parameters with the current DQN's parameter values.
        """
        for target_param, local_param in zip(self.target_dqn.parameters(), self.current_dqn.parameters()):
            target_param.data.copy_(local_param.data)
    
    def _soft_update(self):
        """Soft update model parameters using the formmula: θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(self.target_dqn.parameters(), self.current_dqn.parameters()):
            target_param.data.copy_(self.hp.TAU*local_param.data + (1.0-self.hp.TAU)*target_param.data)

    def _get_max_action_values_of(self, next_states):
        """Computes the max action-value of the next_states. If self.hp.DOUBLE_DQN is set,
        uses the current DQN to determine the max actions, but picks the action values from the target DQN. 

        Args:
            next_states (torch.Tensor): First dimension is batch size (i.e., number of experience tuples)

        Returns:
            torch.Tensor: First dimension is batch size. Second dimension is 1 (only the max action-value is returned)
        """
        if self.hp.DOUBLE_DQN:
            self.current_dqn.eval()
            max_action_indices = self._current_dqn_forward_pass(next_states).detach().max(dim=1).indices.view(-1,1)
            next_states_max_action_values = self.target_dqn(next_states).gather(1,max_action_indices)
        else:
            next_states_max_action_values = self.target_dqn(next_states).detach().max(dim=1).values.view(-1,1)     
        
        return next_states_max_action_values
            
    def learn(self, experiences):
        """Train the current DQN using the given batch of experience tuples.
        Also updates the target DQN according to the choen policy.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences
        
        # experienced state-actions -> (target DQN, experienced rewards) -> target state-action values
        next_states_max_action_values = self._get_max_action_values_of(next_states)
        label_action_values = rewards + (self.hp.GAMMA * next_states_max_action_values * (1 - dones))
        
        # experienced state-actions -> (current DQN) -> current state-action values
        current_action_values = self.current_dqn(states).gather(1, actions)

        # Calculate los using current and target state-action values
        loss = F.mse_loss(current_action_values, label_action_values)

        # Compute gradients and run a step of SGD
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update the target DQN with the new weights of the current DQN
        if self.hp.HARD_UPDATE and (self.t_step % self.hp.OVERWRITE_EVERY == 0):
            self._hard_update()
        else:     
            self._soft_update()                     
