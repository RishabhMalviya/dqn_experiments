import random


class BaseAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

    def act(self, state, *args, **kwargs):
        """Determine the action based on the given state. This action choice is then sent to the environment, 
        which returns the next_state, reward, and whether or not the episode is done.

        Args:
            state (array-like): the current state
            eps (float, optional): The epsilon for the epsilon-greedy policy. Defaults to 0.

        Returns:
            int: The index of the chosen action
        """
        raise NotImplementedError
    
    def step(self, state, action, reward, next_state, done):
        """Use the experience tuple to update the agent's behaviour (i.e., perform learning)

        Args:
            state (torch.Tensor): The initial state
            action (int): The action chosen by the agent from the initial state
            reward (float): The reward recieved from the action
            next_state (torch.Tensor): The next state after the action was executed 
            done (boolean): Whether or not the action causued the episode to terminate
        """
        raise NotImplementedError
    
    def end(self):
        """Call this once the training ends, to perform any cleanup steps if required.
        """
        raise NotImplementedError
