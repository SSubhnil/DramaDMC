import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
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
        terminated = False
        truncated = False
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


class DMControlVecEnv:
    """
    A vectorized environment wrapper for DM Control environments
    that follows Gymnasium's VecEnv API pattern.
    """

    def __init__(self, env_name, num_envs=1, size=(64, 64), camera=None, seed=None):
        """
        Create a vectorized environment of multiple DMControl environments.

        Args:
            env_name: Name of the environment (e.g., 'dm_control/walker-walk')
            num_envs: Number of parallel environments to run
            size: Image size to render
            camera: Camera ID for rendering
            seed: Random seed
        """
        self.num_envs = num_envs
        self.size = size
        self.camera = camera

        # Create vector environment
        def make_env(idx):
            def _init():
                # Set worker-specific rendering settings to avoid conflicts
                if idx > 0:
                    os.environ['MUJOCO_GL'] = 'osmesa'

                env = DMControl(
                    env_name,
                    size=size,
                    camera=camera,
                    seed=seed + idx if seed is not None else None
                )
                return env

            return _init

        self.vec_env = SyncVectorEnv([make_env(i) for i in range(num_envs)])

        # Set up action and observation spaces
        sample_env = DMControl(env_name, size=size, camera=camera)
        self.action_space = sample_env.action_space
        self.observation_space = sample_env.observation_space
        sample_env.close()

    def reset(self):
        """Reset all environments and return initial observations."""
        obs_list = []
        info_list = []

        for i in range(self.num_envs):
            obs, info = self.vec_env.envs[i].reset()
            obs_list.append(obs)
            info_list.append(info)

        return np.stack(obs_list), info_list

    def step(self, actions):
        """
        Step all environments with the given actions.

        Args:
            actions: Actions to take in each environment, shape (num_envs, action_dim)

        Returns:
            observations, rewards, dones, infos
        """
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        for i in range(self.num_envs):
            obs, reward, done, info = self.vec_env.envs[i].step(actions[i])
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

            # Auto-reset environments that are done
            if done:
                reset_obs, reset_info = self.vec_env.envs[i].reset()
                obs_list[i] = reset_obs
                # Store terminal observation in info dict (gymnasium standard)
                info_list[i]['terminal_observation'] = obs

        return np.stack(obs_list), np.array(reward_list), np.array(done_list), info_list

    def close(self):
        """Close all environments."""
        self.vec_env.close()

