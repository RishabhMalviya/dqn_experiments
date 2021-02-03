import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
from collections import deque
import matplotlib.pyplot as plt


from agents import BaseAgent

    
def run_random_agent(env, brain_name):
    env_info = env.reset(train_mode=False)[brain_name]
    action_size = env.brains[env.brain_names[0]].vector_action_space_size

    while True:
        random_action = np.random.randint(action_size)
        env_info = env.step(random_action)[brain_name]

        next_state, done = env_info.vector_observations[0], env_info.local_done[0]

        if done:
            break


def run_trained_agent(env, brain_name, trained_agent: BaseAgent):
    env_info = env.reset(train_mode=False)[brain_name]
    action_size = env.brains[env.brain_names[0]].vector_action_space_size

    while True:
        state = env_info.vector_observations[0]
        action = trained_agent.act(state)
        env_info = env.step(action.item())[brain_name]

        reward, done = env_info.rewards[0], env_info.local_done[0]

        if done:
            break


class TrainingHyperparameters:
    def __init__(self):
        self.EPS_START = 1.0
        self.EPS_END = 0.01
        self.EPS_DECAY = 0.995
        
    def __str__(self):
        return(
            f'TRAINING HYPERPARAMETERS:\n'
            f'\n'
            f'Epsilon (Exploration vs Exploitation):\n'
            f'=========================================\n'           
            f'Starting Epsilon: {self.EPS_START}\n'
            f'Epsilon Lower Limit: {self.EPS_END}\n'
            f'Epsilon Decay: {self.EPS_DECAY}\n'
            f'\n'
            f'\n'
        )


def _plot_average_scores(average_scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(average_scores)), average_scores)
    plt.ylabel('Last 100 Episodes\' Score Average')
    plt.xlabel('Episode #')
    plt.show()    


def train_agent(
    env=None,    
    brain_name=None,
    n_episodes=2000,
    hp: TrainingHyperparameters=TrainingHyperparameters(),
    agent: BaseAgent=None,
    completion_criteria=None
):
    """Driver function for training the an Agent. 
    Training completes when `completion_criteria` is met.
    Saves trained DQN to `dqn_save_path`.
    
    Params
    ======
        env (OpenAI Gym environment): The environment with which the agent will interact
        n_episodes (int): Maximum number of training episodes
        max_t (int): Maximum number of timesteps per episode
        hp (HyperparameterConfig): Hyperparameters
        dqn_save_path (str): Path to file to save trained dqn weights to
        completion_criteria (lambda): Takes in a list of the last 100 scores, and outputs True/False
    """
    print(hp)
    if agent.hp:
        print(agent.hp)
    
    average_scores = []
    scores_window = deque(maxlen=100)
    
    eps = hp.EPS_START
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        score = 0
        
        while True:
            action = agent.act(state, eps)
            env_info = env.step(action.item())[brain_name]
        
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            
            score += reward
            state = next_state

            if done:
                break

        scores_window.append(score)
        average_scores.append(np.mean(scores_window))

        eps = max(hp.EPS_END, hp.EPS_DECAY*eps)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if completion_criteria(scores_window):
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            agent.end()
            break

    _plot_average_scores(average_scores)    
    