# Results
## Random Agent:
![lunarlander-random-agent](https://github.com/RishabhMalviya/dqn_experiments/blob/master/lunar-lander/videos/random_agent.gif?raw=true)
## Trained Agent:
![lunarlander-trained-agent](https://github.com/RishabhMalviya/dqn_experiments/blob/master/lunar-lander/videos/trained_agent.gif?raw=true)

# DQN Approach
While doing soft updates, increasing the rate of the update from 1e-3 to 1e-2 drastically sped up the convergence of the algorithm.

Surprisingly, using a Dueling DQN didn't help speed up convergence at all. I suspect that this is because Dueling DQNs tackle a different kind of problem. They are useful when most of your states don't require any action to be taken (this is a point that was made repeatedly in the paper itself); that simply isn't the case in the Lunar Lander environment.