# Quadcopter Controller based on Deep Reinforcement Learning

The aim of this project is using a DRL approach in order to complete two tasks with 4-engines drone.
Mainly we focused on Hovering and Movement:
- **Hovering**: holding the initial position. In this case we compared a drone fully controlled by the NN with a PID completely tuned using DRL.
- **Movement**: moving the drone to a desired position. In this case the PID comparison did not provide any useful result.

## Premises and Tools

We employed the physics engine PyBullet and the set of RL algorithm from Stable Baseline, starting with solid background frameworks. 
Our target is showing that even a complex phisics system can be controlled in a fully **model-free** way, getting rid of the expert. Also if the code has to be deployed on a micro-controller, not able to run a Neural Network, the DRL approach provides the means to tune a PID without Control Theory knowledge. 
On the other hand the *very* tough work is generalize and abstract the problems without any domain prior. Finding the right combination of:
- **DRL algorithm**: PPO shows a very good trade-off between results and computing time, but also other algorithms have been tried (like A2C).
- **Training duration and phases**: in some cases, long training times are needed before even noticing an improvement in the loss. Also hybridating different reward functions in more training phases has been beneficial.
- **Reward function**: leading the model to achieve the desisered objective requires an heavy effort of handcrafted mathematical reward functions with many points of choices (basically NP-Hard)
- **Network structure**: starting from an Actor-Critic paradigm, we had to choose the depth of the two networks, the width of each layer and the common netowork part between the two flows

## Experiments

Limited by the Home PC performances, we have tried to meaningful explore the possibility for hyper parameters tuning and for the neural architecture structuring.

We have chosen *Proximal Policy Optimization* (PPO) since, according to OpenAI, approximates the state-of-the-art while being simple to implement and to tune. Other attempts, for exemple using A2C, have shown a strong inconsistency in results and a very sensitive response to small perturbation of the hyper parameters.

## Results

|              | No Noise             | White Noise (on all Observation vector) | Selective Noise |
|--------------|----------------------|-----------------------------------------|-----------------|
| Hovering PID | :heavy_check_mark:   | :heavy_check_mark:                      | :x:             |
| Hovering     | :heavy_check_mark:   | :heavy_check_mark:                      | :x:             |
| Movement     | :heavy_check_mark:   | :heavy_check_mark:                      | :x:             |