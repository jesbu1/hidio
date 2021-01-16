from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.num_timesteps = 0
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/pusher.xml' % dir_path, 4)
        utils.EzPickle.__init__(self)
        self.reset_model()

    def step(self, a):
        self.num_timesteps += 1
        self.do_simulation(a, self.frame_skip)
        obj_pos = self.get_body_com("object"),
        vec_1 = obj_pos - self.get_body_com("tips_arm")
        vec_2 = obj_pos - self.get_body_com("goal")

        #reward_near = -np.sum(np.abs(vec_1))
        #reward_dist = -np.sum(np.abs(vec_2))
        reward_ctrl = 0.001 * -np.square(a).sum()
        #reward_ctrl = 0
        #reward = 1.25 * reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
       
        success = False
        if np.sqrt(np.sum(np.square(vec_2))) <= 0.25:
            success = True
        ob = self._get_obs()
        return ob, float(success) + reward_ctrl, self.num_timesteps >= 100, {'is_success': success}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0]) 
        self.cylinder_pos = np.array([-0.25, 0.15]) + np.random.normal(0, 0.025, [2])

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        self.ac_goal_pos = self.get_body_com("goal")

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[:7],
            self.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
        ])

    def reset(self):
        self.num_timesteps = 0
        return super().reset()
        
if __name__ == '__main__':
    env = PusherEnv()
    done = False
    obs = env.reset()
    counter = 0
    import pdb; pdb.set_trace()
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        counter += 1
        print(obs, reward, done, info)
    print(counter)