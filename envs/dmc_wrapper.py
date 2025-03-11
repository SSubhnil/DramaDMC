# In env_wrapper.py
import gymnasium as gym
import numpy as np
from dm_control import suite
from pandas.core.config_init import is_terminal
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
import os

class DMControl(gym.Env):
    def __init__(
            self,
            name,
            action_repeat=4,
            size=(64, 64),
            camera=None,
            seed=None,
    ):
        # Ensure software rendering is used (add this line)
        os.environ["MUJOCO_GL"] = "osmesa"
        # Parse domain and task from name (e.g., "dm_control/cartpole-balance")
        domain_name, task_name = name.split('/')[-1].split('-')

        # Create DM Control environment
        self._env = suite.load(domain_name, task_name, task_kwargs={'random': seed})

        # Wrap with compatibility class
        self._env = DmControlCompatibilityV0(
            self._env,
            render_mode="rgb_array",
            render_height=size[0],
            render_width=size[1],
            camera_id=camera if camera is not None else 0
        )

        self._action_repeat = action_repeat
        self._size = size
        self._seed = seed
        self._episode_steps = 0

        # Set action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._env.action_spec().shape[0],),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(size[0], size[1], 3),
            dtype=np.uint8
        )

    def step(self, action):
        total_reward = 0.0
        self._episode_steps += 1  # Increment step counter

        for _ in range(self._action_repeat):
            _, reward, terminated, truncated, info = self._env.step(action)
            # Get image observation from physics engine
            image = self._env._env.physics.render(
                height=self._size[0],
                width=self._size[1],
                camera_id=self._env.camera_id
            )

            total_reward += reward
            is_terminal = terminated or truncated
            if is_terminal:
                break

        info = {}
        # Return in format expected by your code (4 values)
        info['is_terminal'] = terminated
        info['episode_frame_number'] = self._episode_steps * self._action_repeat

        # Reset episode counter if episode is done
        is_last = terminated or truncated
        if is_last:
            self._episode_steps = 0

        image = np.ascontiguousarray(image)
        return image, total_reward, is_last, False, info


    def reset(self):
        # Reset episode steps counter
        self._episode_steps = 0

        # Original reset returns state-based observation
        _, info = self._env.reset(seed=self._seed)

        # Get image observation
        image = self._env._env.physics.render(
            height=self._size[0],
            width=self._size[1],
            camera_id=self._env.camera_id
        )

        image = np.ascontiguousarray(image)
        return image, info

    def close(self):
        return self._env.close()
