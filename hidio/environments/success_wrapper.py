import gym
import gin 

@gin.configurable
class SuccessWrapper(gym.Wrapper):
    """Retrieve the success info from the environment return.
    """

    def __init__(self, env, since_episode_steps):
        super().__init__(env)
        self._since_episode_steps = since_episode_steps

    def reset(self, **kwargs):
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._steps += 1

        info["success"] = 0.0
        # only count success after certain steps
        if self._steps >= self._since_episode_steps and info["is_success"] == 1:
            info["success"] = 1.0

        info.pop("is_success")  # from gym, we remove it here
        return obs, reward, done, info