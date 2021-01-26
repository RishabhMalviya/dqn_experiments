import imageio
import os
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from dqn_agent import HyperparameterConfig, Agent


def _save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    """Takes a list of frames (each frame can be generated with the `env.render()` function from OpenAI gym)
    and converts it into GIF, and saves it to the specified location.
    Code adapted from this gist: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

    Args:
        frames (list): A list of frames generated with the env.render() function
        path (str, optional): The folder in which to save the generated GIF. Defaults to './'.
        filename (str, optional): The target filename. Defaults to 'gym_animation.gif'.
    """
    imageio.mimwrite(os.path.join(path, filename), frames, fps=60)


def save_random_agent_gif(env):
    frames = []
    for i in range(5):
        state = env.reset()
        for t in range(500):
            action = env.action_space.sample()
            frames.append(env.render(mode="rgb_array"))
            state, _, done, _ = env.step(action)
            if done:
                break

    env.close()

    _save_frames_as_gif(frames, path='./videos', filename='random_agent.gif')


def save_trained_agent_gif(env, agent, dqn_save_path='trained_dqn.pth'):
    agent.current_dqn.load_state_dict(torch.load(dqn_save_path))

    frames = []
    for i in range(5):
        state = env.reset()
        for t in range(500):
            action = agent.act(state)
            frames.append(env.render(mode="rgb_array"))
            state, _, done, _ = env.step(action)
            if done:
                break
            
    env.close()

    _save_frames_as_gif(frames, path='./videos/', filename='trained_agent.gif')


def train_agent(
    env=None,    
    n_episodes=2000,
    max_t=1500,
    hp: HyperparameterConfig=None,
    agent: Agent = None,
    dqn_save_path='./trained_dqn.pth',
    completion_criteria=None
):
    """Driver function for training the DQN Agent. 
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
    if not hp:
        hp = HyperparameterConfig()
    print(hp)
        
    average_scores = []
    scores_window = deque(maxlen=100)
    eps = hp.EPS_START
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
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
            torch.save(agent.current_dqn.state_dict(), 'trained_dqn.pth')
            break

    # Plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(average_scores)), average_scores)
    plt.ylabel('Last 100 Episodes\' Score Average')
    plt.xlabel('Episode #')
    plt.show()
