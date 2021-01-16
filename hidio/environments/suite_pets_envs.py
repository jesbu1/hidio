# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Suite for loading OpenAI Robotics environments.

**NOTE**: Mujoco requires separated installation.

(gym >= 0.10, and mujoco>=1.50)

Follow the instructions at:

https://github.com/openai/mujoco-py

"""

try:
    import mujoco_py
except ImportError as e:
    mujoco_py = None
    
from hidio.environments.pets_envs.pusher import PusherEnv
from hidio.environments.pets_envs.reacher import Reacher3DEnv

import functools
import numpy as np
import gym

import gin
from alf.environments import suite_gym, alf_wrappers, process_environment
from alf.environments.utils import UnwrappedEnvChecker
from .success_wrapper import SuccessWrapper

_unwrapped_env_checker_ = UnwrappedEnvChecker()


def is_available():
    return mujoco_py is not None 

@gin.configurable
class PETSSuccessWrapper(SuccessWrapper):
    """Retrieve the success info from the environment return.
    """

    def __init__(self, env, since_episode_steps):
        super().__init__(env=env, since_episode_steps=since_episode_steps)
        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            #'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

class ActionScalingWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        high = self.action_space.high
        low = self.action_space.low
        assert np.all(np.abs(low) == high)
        self._magnitude = high
        # scale the action space to (-1, 1) for better entropy calc
        self.action_space.high = np.ones_like(high)
        self.action_space.low = np.ones_like(high) * -1

    def step(self, action):
        # scale action back to its original magnitude
        action *= self._magnitude
        return self.env.step(action)

@gin.configurable
def load(env_name,
         env_id=None,
         discount=1.0,
         max_episode_steps=None,
         use_success_wrapper=True,
         gym_env_wrappers=(),
         alf_env_wrappers=(),
         wrap_with_process=False):
    """Loads the selected environment and wraps it with the specified wrappers.

    Note that by default a ``TimeLimit`` wrapper is used to limit episode lengths
    to the default benchmarks defined by the registered environments.

    Args:
        env_name: Ignored, but required for create_environment in utils.py
        discount: Discount to use for the environment.
        max_episode_steps: If None the ``max_episode_steps`` will be set to the default
            step limit defined in the environment's spec. No limit is applied if set
            to 0 or if there is no ``timestep_limit`` set in the environment's spec.
        gym_env_wrappers: Iterable with references to wrapper classes to use
            directly on the gym environment.
        alf_env_wrappers: Iterable with references to wrapper classes to use on
            the torch environment.

    Returns:
        An AlfEnvironment instance.
    """
    _unwrapped_env_checker_.check_and_update(wrap_with_process)


    def env_ctor(env_id=None):
        return suite_gym.wrap_env(
            env,
            env_id=env_id,
            discount=discount,
            max_episode_steps=max_episode_steps,
            gym_env_wrappers=gym_env_wrappers,
            alf_env_wrappers=alf_env_wrappers)
    if env_name == "Pusher":
        env = PusherEnv()
    elif env_name == "Reacher":
        env = Reacher3DEnv()
    env = ActionScalingWrapper(env)
    if use_success_wrapper:
        env = PETSSuccessWrapper(env, max_episode_steps)

    if wrap_with_process:
        process_env = process_environment.ProcessEnvironment(
            functools.partial(env_ctor))
        process_env.start()
        torch_env = alf_wrappers.AlfEnvironmentBaseWrapper(process_env)
    else:
        torch_env = env_ctor(env_id=env_id)

    return torch_env

