from gym.envs.registration import register

register(
    id="Sparrow-v0",
    entry_point="SparrowV0.envs:SparrowV0Env",
    max_episode_steps=2000
)
