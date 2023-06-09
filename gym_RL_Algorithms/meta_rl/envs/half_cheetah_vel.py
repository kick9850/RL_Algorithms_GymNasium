from typing import Any, Dict, List, Tuple

import numpy as np

from gym_RL_Algorithms.meta_rl.envs import register_env
from gym_RL_Algorithms.meta_rl.envs.half_cheetah import HalfCheetahEnv


@register_env("cheetah-vel")
class HalfCheetahVelEnv(HalfCheetahEnv):
    def __init__(self, num_tasks: int) -> None:
        self.tasks = self.sample_tasks(num_tasks)
        self._task = self.tasks[0]
        self._goal_vel = self._task["velocity"]
        super().__init__()
        self._reset_noise_scale = 0.1

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, Dict[str, Any]]:
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]

        progress = (xposafter - xposbefore) / self.dt
        run_cost = progress - self._goal_vel
        scaled_run_cost = -1.0 * abs(run_cost)
        control_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = scaled_run_cost - control_cost
        done = False
        info = dict(run_cost=run_cost, control_cost=-control_cost, task=self._task)
        return observation, reward, done, info

    def sample_tasks(self, num_tasks: int):
        np.random.seed(0)
        velocities = np.random.uniform(0.0, 2.0, size=(num_tasks,))
        tasks = [{"velocity": velocity} for velocity in velocities]
        return tasks

    def get_all_task_idx(self) -> List[int]:
        return list(range(len(self.tasks)))

    def reset_task(self, idx: int) -> None:
        self._task = self.tasks[idx]
        self._goal_vel = self._task["velocity"]
        self.reset()

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
                self.init_qvel
                + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation