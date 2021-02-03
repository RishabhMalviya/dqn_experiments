# Overview
This repository contains various implementation of DQNs, with a number of additional tricks that have been propsed since then (see [Bibliography](#bibilography)). I have applied these DQNs to various OpenAI Gym environments.


# Results
These are the results (before and after GIFs) for the experiments that I have run.

| **Banana Collector** (DQN) ||
|--------|--------|
|*Before Training*|*After Training*|
|![banana-collector-random-agent](https://github.com/RishabhMalviya/dqn_experiments/blob/master/banana-collector/videos/random_agent.gif?raw=true)|![banana-collector-trained-agent](https://github.com/RishabhMalviya/dqn_experiments/blob/master/banana-collector/videos/trained_agent.gif?raw=true)|

| **Lunar Lander v2** (DQN) ||
|--------|--------|
|*Before Training*|*After Training*|
|![lunarlander-random-agent](https://github.com/RishabhMalviya/dqn_experiments/blob/master/lunar-lander/videos/random_agent.gif?raw=true)|![lunarlander-trained-agent](https://github.com/RishabhMalviya/dqn_experiments/blob/master/lunar-lander/videos/trained_agent.gif?raw=true)|

| **Cart Pole v1** (Dueling DQN) ||
|--------|--------|
|*Before Training*|*After Training*|
|![cartpole-random-agent](https://github.com/RishabhMalviya/dqn_experiments/blob/master/cart-pole/videos/random_agent.gif?raw=true)|![cartpole-trained-agent](https://github.com/RishabhMalviya/dqn_experiments/blob/master/cart-pole/videos/trained_agent.gif?raw=true)|


# Local Setup
This setup was done on a system with these specifications:
1. **OS**: Windows 10
2. **CUDA Toolkit Version**: 11.2 (Download it from [here](https://developer.nvidia.com/Cuda-downloads))
3. **Python Version**: Python 3.6.8 (You can download the executable installer for Windows 10 from [here](https://www.python.org/ftp/python/3.6.8/python-3.6.8-amd64.exe))
4. **Unity ML Agents (Udacity Version)**: A managed build of Unity ML Agents framework, which can be found in the folder `unity-ml-agents-setup`; you just have to go into the folder and run `pip install .`. This only needs to be installed for the environment `banana-collector`.

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
7. Run `cd ./unity-ml-agents-setup` and `pip install .` also if you plan on working with Unity ML Agents or running the notebook in `banana-collector`. 
8. Finally, start a Jupyter Notebook (run `jupyter notebook` from Powershell) from the root of the repo and hack away!


# User Guide - Quickstart

## Basic Abstractions

At a minimum, the following four objects need to be instantiated for running an experiment on any of the environments:

### 1. The Environment
This will be an [OpenAI gym enviroment](https://gym.openai.com/docs/).

Make sure you've gone through the steps in [Local Setup](#local-setup), if you want to use the [Box 2D environments](https://gym.openai.com/envs/#box2d).

### 2. The Agent
Depending on which agent you are using, you may have to instantiate additional 'sub-objects'. For example, the `DQNAgent` requires a `DQN` (`torch nn.Module`) and an optional `DQNHyperparameters` object during initialization.

Further details can be found in the agents' [`README.md`](https://github.com/RishabhMalviya/dqn_experiments/tree/master/agents/README.md)

### 3. Training Hyperparameters
The hyperparameters used during training are defined in the `TrainingHyperparameters` class in the `train_and_visualize.py` file. These are the defaults:
```
self.EPS_START = 1.0
self.EPS_END = 0.01
self.EPS_DECAY = 0.995
```

## Training
The agent can then be set free to interact with environment and learn using the `train_agent` function from the `train_and_visualize.py` file. This function takes an argument called `completion_criteria`, which is supposed to be a function that takes as an argument a list of the scores from the last 100 episodes (latest first), and returns True or False. For example:
```
train_agent(   
    env=env,
    agent=agent,
    n_episodes=2000,
    max_t=1500,
    completion_criteria=lambda scores_window: np.mean(scores_window) >= 200.0
)
```

For examples of this in action, go into the folders that are named with an environment name, for example, [`lunar-lander`](https://github.com/RishabhMalviya/dqn_experiments/tree/master/lunar-lander) and explore the Jupyter Notebooks therein. 

Those codes were the ones used to generate the GIFs you saw at the beginning of this README.md

## Visualizing
You can visualize the trained agent (or a randomly behaving agent) and save GIFs of the interaction with the functions `save_trained_agent_gif` (or `save_random_agent_gif`) in `train_and_visualize.py`.

And that's it! If you face any problems, or have any questions. please add an Issue to the repo.

# Bibilography

1. [Human-Level Control Through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) - *Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg & Demis Hassabis*
2. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf) - *Hado van Hasselt and Arthur Guez and David Silver*
3. [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf) - *Ziyu Wang and Tom Schaul and Matteo Hessel and Hado van Hasselt and Marc Lanctot and Nando de Freitas*