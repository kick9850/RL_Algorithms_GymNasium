import numpy as np
import gymnasium
from gymnasium import utils
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv as HalfCheetahEnv_
from gymnasium.envs.mujoco import mujoco_env


class HalfCheetahEnv(HalfCheetahEnv_, mujoco_env.MujocoEnv):
    def __init__(self):
        observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (18,), np.float64)
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5, observation_space)
        utils.EzPickle.__init__(self)

    def _get_obs(self) -> np.ndarray:
        return (
            np.concatenate(
                [
                    self.data.qpos.flat[1:],
                    self.data.qvel.flat,
                    self.get_body_com("torso").flat,
                ],
            )
            .astype(np.float32)
            .flatten()
        )

    def viewer_setup(self) -> None:
        camera_id = self.model.camera_name2id("track")
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        self.viewer._hide_overlay = True