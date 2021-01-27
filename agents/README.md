# The Agents

## DQNAgent
This agent and its associated classes are defined in `dqn_agent.py`. 

### Tricks
The agent has a number of tricks implemented in it (see [Bibliography](#bibilography)):
1. *Soft Update* - instead of updating the target DQN with the current DQN only after fixed intervals of steps, update it 'softly' with a weighted average of the target DQN and the current DQN at every step.
   1. The weight of the current DQN during the soft update is set with `TAU`. 
   2. You can also choose not to do soft updates by setting `HARD_UPDATE` to `True`.
   3. If you're doing hard updates, you'll need to set the interval of steps after which you want to perform the hard update with `OVERWRITE_STEPS`.
2. *Double DQN* - The loss used when training the current DQN is the square of the following:
![loss function](https://imgur.com/bKrBclq.jpg)
You will recognize the first two terms from the Bellman Equations. The Double DQN modifies the process of determining this quantity. The highest valued actions are determined using the current DQN, but the action-value itself is calculated with the target DQN.
   1. You can switch on Double DQN by setting `DOUBLE_DQN` to `True`.

### Hyperparameters
The default hyperparameters are defined in the `DQNHyperparameters` class (in the same file, `agents/dqn_agent.py`):

```
self.BUFFER_SIZE = int(1e5)  # replay buffer size
self.GAMMA = 0.99            # discount factor

self.BATCH_SIZE = 64         # minibatch size
self.LR = 5e-4               # learning rate 
self.UPDATE_EVERY = 4        # how often to the current DQN should learn

self.HARD_UPDATE = False     # to hard update or not (with double DQN, one should)
self.DOUBLE_DQN = False      # to use double DQN or not
self.TAU = 1e-3              # for soft update of target parameters
self.OVERWRITE_EVERY = 128   # how often to clone the local DQN to the target DQN
```

### DQN
The architectures for the DQN that the `DQNAgent` will use needs to be defined by the user. These may contain an implementation of a *DuelingDQN*.


# Bibilography

1. [Human-Level Control Through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) - *Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg & Demis Hassabis*
2. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf) - *Hado van Hasselt and Arthur Guez and David Silver*
3. [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf) - *Ziyu Wang and Tom Schaul and Matteo Hessel and Hado van Hasselt and Marc Lanctot and Nando de Freitas*