from gym.envs.registration import register

register(
    id="gym_tankwar/TankWar-v0",
    entry_point="gym_tankwar.envs:TankWar",
)
