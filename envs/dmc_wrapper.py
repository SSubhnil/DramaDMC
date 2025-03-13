import gymnasium as gym
import numpy as np
from dm_control import suite
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
        # Parse domain and task from name (e.g., "dm_control/cartpole-balance")
        domain_name, task_name = name.split('/')[-1].split('-')

        # Configure rendering mode based on worker index
        worker_id = os.getenv('WORKER_ID', '0')
        # Only worker 0 renders with EGL, others use CPU rendering
        if worker_id != '0':
            os.environ['MUJOCO_GL'] = 'osmesa'

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

        self.episode_steps = 0

        # Set action and observation spaces
        self.action_space = self._env.action_space
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(size[0], size[1], 3),
            dtype=np.uint8
        )

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._action_repeat):
            _, reward, terminated, truncated, info = self._env.step(action)

            # Get image observation instead
            image = self._env.physics.render(
                height=self._size[0],
                width=self._size[1],
                camera_id=self._env.camera_id
            )

            total_reward += reward
            if terminated or truncated:
                break

        # Return in format expected by your code (4 values)
        is_last = terminated or truncated
        info['is_terminal'] = terminated  # Store termination state in info
        image = np.ascontiguousarray(image)

        self.episode_steps += 1
        info['episode_frame_number'] = self.episode_steps
        return image, total_reward, is_last, info

    def reset(self):
        # Original reset returns state-based observation
        _, info = self._env.reset(seed=self._seed)

        # Get image observation instead
        image = self._env.physics.render(
            height=self._size[0],
            width=self._size[1],
            camera_id=self._env.camera_id
        )
        image = np.ascontiguousarray(image)
        self.episode_steps = 0
        return image, info

    def close(self):
        return self._env.close()
