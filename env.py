import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BarberEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        
        # 3.1 New XML path
        XML_PATH = "assets/so101_barber_real.xml"
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        
        # Find hair geometry ID for hiding
        self.hair_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "hair")
        
        # 3.2 Joint list & action space (SO-101 uses joint names "1"‥"6")
        JOINTS = ["1", "2", "3", "4", "5", "6"]
        self.JID = [self.model.joint(n).id for n in JOINTS]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        
        # 3.3 Action → target mapping
        self.low = self.model.jnt_range[self.JID, 0]
        self.high = self.model.jnt_range[self.JID, 1]
        
        # Observation space: joint positions + velocities + sensor data
        obs_dim = len(self.data.qpos) + len(self.data.qvel) + len(self.data.sensordata)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        
        # 3.4 Sensor index
        self.CUT_ADR = self.model.sensor("cut_hair").adr
        
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
        # 3.3 Control mapping: scale [-1,1] → joint range
        target = self.low + 0.5 * (action + 1.0) * (self.high - self.low)
        self.data.ctrl[:] = target  # position actuators already exist
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Check if blade touches hair
        reward = 0.0
        if self.data.sensordata[self.CUT_ADR] > 0:
            # Hide hair when cut
            self.model.geom_rgba[self.hair_gid, 3] = 0.0  # Make hair invisible
            reward = 1.0  # Reward for successful cut
        
        obs = self._get_obs()
        terminated = False
        truncated = False
        info = {"hair_cut": self.data.sensordata[self.CUT_ADR] > 0}
        
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