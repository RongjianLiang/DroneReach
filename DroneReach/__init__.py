from gymnasium.envs.registration import register

register(
     id="DroneReachDiscrete",
     entry_point="DroneReach.envs:DroneReachDiscrete",
     max_episode_steps=50,
)