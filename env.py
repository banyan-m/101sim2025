import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BarberEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path("assets/so101_barber.xml")
        self.data = mujoco.MjData(self.model)
        
        # Find hair geometry ID for hiding
        self.hair_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "hair")
        
        # Action space: 4 joint motors
        self.action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        
        # Observation space: joint positions + velocities + sensor data
        obs_dim = len(self.data.qpos) + len(self.data.qvel) + len(self.data.sensordata)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        
        # Setup rendering if needed
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset hair visibility
        self.model.geom_rgba[self.hair_gid, 3] = 1.0  # Make hair visible again
        
        return self._get_obs(), {}
    
    def step(self, action):
        # Apply actions to motors (scale to reasonable torque range)
        max_torque = 1.0  # Reduced from 2.0 for stability
        self.data.ctrl[:] = action * max_torque
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Check if clipper touches hair
        reward = 0.0
        if len(self.data.sensordata) > 0 and self.data.sensordata[0] > 0:
            # Hide hair when cut
            self.model.geom_rgba[self.hair_gid, 3] = 0.0  # Make hair invisible
            reward = 1.0  # Reward for successful cut
        
        obs = self._get_obs()
        terminated = False
        truncated = False
        info = {"hair_cut": self.data.sensordata[0] > 0 if len(self.data.sensordata) > 0 else False}
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        obs_parts = [self.data.qpos, self.data.qvel]
        if len(self.data.sensordata) > 0:
            obs_parts.append(self.data.sensordata)
        return np.concatenate(obs_parts).astype(np.float32)
    
    def render(self):
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        else:
            return None 