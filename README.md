# Overview
This repository contains an implementation of DQNs, with a number of additional tricks that have been propsed since then (see Biobliography). I have applied these DQNs to various OpenAI Gym environments.

# Results
These are the results (before and after GIFs) for the experiments that I have run. Corresponding entries to the OpenAI leaderboards are also linked to.

## LunarLander-v2
[Leaderboard Link](https://github.com/openai/gym/wiki/Leaderboard#lunarlander-v2)
### Before Training:
![lunarlander-random-agent](https://github.com/RishabhMalviya/dqn_experiments/blob/master/lunar-lander/videos/random_agent.gif?raw=true)
### After Training:
![lunarlander-trained-agent](https://github.com/RishabhMalviya/dqn_experiments/blob/master/lunar-lander/videos/trained_agent.gif?raw=true)

# Local Setup
This setup was done on a system with these specifications:
1. **OS**: Windows 10
2. **CUDA Toolkit Version**: 11.2 (Download it from [here](https://developer.nvidia.com/Cuda-downloads))
3. **Python Version**: Python 3.6.8 (You can download the executable installer for Windows 10 from [here](https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe))
But I've kept things very simple and put in lots of links, so as to make it easier for you to figure out the corresponding steps for Linux.

Here are the exact steps:
1. Clone this repository with (use [Git Bash for Windows](https://gitforwindows.org/)) `git clone https://github.com/RishabhMalviya/dqn-experiments.git`.
2. In the cloned repository, create a venv by running the following command from Powershell (venv should be installed along with the Python 3.6.8 installation form the link given above): `python -m venv ./venv`. Also, in case you're using Anaconda, you should launch Powershell by searching for "*Anaconda Powershell Prompt*" from Start.
3. We can't `pip install` the `requirements.txt` just yet because one of the dependencies, `box2d-py`, requires a build tool called Swig, which we'll have to install first: 
   1. As the [Windows documentation for Swig](http://www.swig.org/Doc1.3/Windows.html) says, *Download the [swigwin zip package](http://prdownloads.sourceforge.net/swig/swigwin-4.0.2.zip) from the [SWIG website](http://www.swig.org/download.html) and unzip into a directory. This is all that needs downloading for the Windows platform.* Note that the installation directory needs to be in your `PATH` environment variable.
   2. To get Swig to build `box2d-py` correctly, you will also have to set the following two environment variables. Change `</path/to/python>` to correspond to the python with which you created the venv:
      1. `PYTHON_INCLUDE`: `</path/to/python>/include`
      2. `PYTHON_LIB`: `</path/to/python>/libs/python36.lib`
4. Now, activate the venv by running `./venv/Scripts/activate` in Powershell.
5. Upgrade pip with `pip install -U pip`.
6. And install the requirements with `pip install -r requirements.txt`. You should adapt the first three lines of the `requirements.txt` file based on the installation command that the [PyTorch download page](https://pytorch.org/get-started/locally/) recommends for your system.
7. Finally, start a Jupyter Notebook (run `jupyter notebook` from Powershell) from the root of the repo and hack away!

# Code Internals

Four objects need to be instantiated for running an experiment on any of the environments:

1. **The Environment**
2. **The Agent**
   1. *Hyperparameters*
   2. *DQN* 

## Hyperparameters
The default hyperparameters are defined in the HyperparameterConfig class in the file dqn_agent.py in the parent directory:

```
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
```

## Agent
The agent itself is defined in the `Agent` class in the file `dqn_agent.py` in the root folder. This agent has a number of tricks implemented in it including (each trick is followed by a list of the corresponding configs in the `HyperparameterConfig` class):
1. *Soft Update* - instead of updating the target DQN after fixed intervals of steps, update it 'softly' with a weighted average of the target DQN and the current DQN.
   1. The weight of the current DQN during the soft update is set with `TAU`. 
   2. You can also choose not to do soft updates by setting `HARD_UPDATE` to `True`.
   3. If you're doing hard updates, you'll need to set the interval of steps after which you want to perform the hard update with `OVERWRITE_STEPS`.
2. *Double DQN* - The loss used when training the current DQN is the square of the following:
![loss function](https://imgur.com/bKrBclq.jpg)
You will recognize the first two terms from the Bellman Equations. The Double DQN modifies the process of determining this quantity. The highest valued actions are determined using the current DQN (the parameters \theta), then the action-value itself is taken from the target DQN (parameters \theta^-).
   1. You can switch on Double DQN by setting `DOUBLE_DQN` to `True`.

## DQN
The architecture of the DQNs that the agent internally instantiates are defined in each of the environments' folder in a file called `model.py`.