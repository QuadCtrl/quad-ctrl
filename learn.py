"""Script demonstrating the use of `gym_pybullet_drones`' Gym interface.

Class TakeoffAviary is used as a learning env for the A2C and PPO algorithms.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
The boolean argument --rllib switches between `stable-baselines3` and `ray[rllib]`.
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning libraries `stable-baselines3` and `ray[rllib]`.
It is not meant as a good/effective learning example.

"""
import sys

sys.path.append('../')
import time
import argparse
import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseCustom import BaseCustom
from gym_pybullet_drones.utils.utils import sync, str2bool


class Model:
    def __init__(self):
        self.states = 0

    def predict(self, obs, deterministic=True):
        action = -1 * np.array([[1, 1, 1, 1]])
        return action, self.states


if __name__ == "__main__":

    #### Check the environment's spaces ########################
    # env = gym.make("takeoff-aviary-v0")
    # print("[INFO] Action space:", env.action_space)
    # print("[INFO] Observation space:", env.observation_space)
    # check_env(env, warn=True, skip_render_check=True)

    env = BaseCustom(gui=True, record=False, initial_xyzs=np.array([[2, 2, 2]]))
    check_env(env, warn=True, skip_render_check=True)

    # TODO: Train and plot iterations

    #### Show (and record a video of) the model's performance ##
    # env = HoverAviary(gui=True, record=False)
    logger = Logger(logging_freq_hz=int(env.SIM_FREQ / env.AGGR_PHY_STEPS), num_drones=1)
    obs = env.reset()
    start = time.time()
    model = Model()

    for i in range(3*env.SIM_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        logger.log(drone=0,
                   timestamp=i / env.SIM_FREQ,
                   state=np.hstack([obs[0:3], np.zeros(4), obs[3:15], np.resize(action, (4))]),
                   control=np.zeros(12)
                   )
        if i % env.SIM_FREQ == 0:
            env.render()
            print(done)
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()

    env.close()
    logger.plot()
