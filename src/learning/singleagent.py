"""Learning script for single agent problems.

Agents are based on `stable_baselines3`'s implementation of A2C, PPO SAC, TD3, DDPG.

Example
-------
To run the script, type in a terminal:

    $ python singleagent.py --env <env> --algo <alg> --obs <ObservationType> --act <ActionType> --cpu <cpu_num>

Notes
-----
Use:

    $ tensorboard --logdir ./results/save-<env>-<algo>-<obs>-<act>-<time-date>/tb/

To check the tensorboard results at:

    http://localhost:6006/

"""
import sys

sys.path.append('../../')

import os
import time
from datetime import datetime
from sys import platform
import argparse
import subprocess
import numpy as np
import gym
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import \
    make_vec_env  # Module cmd_util will be renamed to env_util https://github.com/DLR-RM/stable-baselines3/pull/197
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import A2C, PPO, SAC, TD3, DDPG
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold

from gym_pybullet_drones.envs.single_agent_rl.MoveAviary import MoveAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

import shared_constants

EPISODE_REWARD_THRESHOLD = -0  # Upperbound: rewards are always negative, but non-zero
"""float: Reward threshold to halt the script."""


def check_parser(ARGS):
    #### Warning ###############################################
    if ARGS.act == ActionType.ONE_D_RPM or ARGS.act == ActionType.ONE_D_DYN or ARGS.act == ActionType.ONE_D_PID:
        print("\n\n\n[WARNING] Simplified 1D problem for debugging purposes\n\n\n")
        #### Errors ################################################
        if not ARGS.env in ['hover']:
            print("[ERROR] 1D action space is only compatible with HoverAviary")
            exit()
    if ARGS.algo in ['sac', 'td3', 'ddpg'] and ARGS.cpu != 1:
        print("[ERROR] The selected algorithm does not support multiple environments")
        exit()


if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning experiments script')
    parser.add_argument('--env', default='move', type=str, choices=['move', 'hover'], help='Task (default: hover)',
                        metavar='')
    parser.add_argument('--algo', default='ppo', type=str, choices=['a2c', 'ppo', 'sac', 'td3', 'ddpg'],
                        help='RL agent (default: ppo)', metavar='')
    parser.add_argument('--obs', default='kin', type=ObservationType, help='Observation space (default: kin)',
                        metavar='')
    parser.add_argument('--act', default='rpm', type=ActionType, help='Action space (default: one_d_rpm)', metavar='')
    parser.add_argument('--cpu', default='1', type=int, help='Number of training environments (default: 1)', metavar='')
    parser.add_argument('--resume', default='results/save-move-ppo-kin-rpm-03.01.2022_19.37.27/success_model.zip',
                        type=str, help='Resume Training from file path', metavar='')

    ARGS = parser.parse_args()
    check_parser(ARGS)

    #### Save directory ####

    filename = os.path.dirname(os.path.abspath(
        __file__)) + '/results/save-' + ARGS.env + '-' + ARGS.algo + '-' + ARGS.obs.value + '-' + ARGS.act.value + '-' + datetime.now().strftime(
        "%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename + '/')

    env_name = ARGS.env + "-aviary-v0"

    if env_name == "move-aviary-v0":
        train_env = MoveAviary(initial_xyzs=np.array([[0, 0, 1]]),
                               aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                               obs=ARGS.obs,
                               act=ARGS.act)
    if env_name == "hover-aviary-v0":
        train_env = HoverAviary(initial_xyzs=np.array([[0, 0, 1]]),
                                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                obs=ARGS.obs,
                                act=ARGS.act)

    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)

    #### On-policy algorithms ##################################
    onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                           net_arch=[128, 128, dict(vf=[256], pi=[256])]
                           # net_arch=[128, 128, dict(vf=[256], pi=[128, 256])]
                           )
    if ARGS.algo == 'a2c':
        model = A2C(a2cppoMlpPolicy,
                    train_env,
                    policy_kwargs=onpolicy_kwargs,
                    tensorboard_log=filename + '/tb/',
                    verbose=1,
                    seed=0
                    )
    if ARGS.algo == 'ppo':
        model = PPO(a2cppoMlpPolicy,
                    train_env,
                    policy_kwargs=onpolicy_kwargs,
                    tensorboard_log=filename + '/tb/',
                    verbose=1,
                    seed=0
                    )

    #### Off-policy algorithms #################################
    offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                            net_arch=[128, 128, 128, 256]
                            )  # or None # or dict(net_arch=dict(qf=[256, 128, 64, 32], pi=[256, 128, 64, 32]))
    if ARGS.algo == 'sac':
        model = SAC(sacMlpPolicy,
                    train_env,
                    policy_kwargs=offpolicy_kwargs,
                    tensorboard_log=filename + '/tb/',
                    verbose=1
                    )
    if ARGS.algo == 'td3':
        model = TD3(td3ddpgMlpPolicy,
                    train_env,
                    policy_kwargs=offpolicy_kwargs,
                    tensorboard_log=filename + '/tb/',
                    verbose=1,
                    )
    if ARGS.algo == 'ddpg':
        model = DDPG(td3ddpgMlpPolicy,
                     train_env,
                     policy_kwargs=offpolicy_kwargs,
                     tensorboard_log=filename + '/tb/',
                     verbose=1
                     )

    #### Create eveluation environment #########################
    if ARGS.obs == ObservationType.KIN:
        eval_env = gym.make(env_name,
                            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                            obs=ARGS.obs,
                            act=ARGS.act
                            )

    #### Train the model #######################################
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=EPISODE_REWARD_THRESHOLD,
                                                     verbose=1
                                                     )
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename + '/',
                                 log_path=filename + '/',
                                 eval_freq=int(2000 / ARGS.cpu),
                                 deterministic=True,
                                 render=False
                                 )

    # RESUMES PREVIOUSLY TRAINED MODEL
    if ARGS.resume:
        path_prev_train = ARGS.resume
        if 'ppo' in path_prev_train:
            model = PPO.load(path_prev_train)
        elif 'a2c' in path_prev_train:
            model = A2C.load(path_prev_train)
        elif 'ddpg' in path_prev_train:
            model = DDPG.load(path_prev_train)
        elif 'td3' in path_prev_train:
            model = TD3.load(path_prev_train)
        elif 'sac' in path_prev_train:
            model = SAC.load(path_prev_train)
        model.set_env(eval_env)

    model.learn(total_timesteps=600000,  # int(1e12),
                callback=eval_callback,
                log_interval=100,
                )

    #### Save the model ########################################
    model.save(filename + '/success_model.zip')
    print(filename)
