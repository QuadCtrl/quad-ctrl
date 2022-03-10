from gym.envs.registration import register


register(
    id='move-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl:MoveAviary',
)

register(
    id='hover-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl:HoverAviary',
)