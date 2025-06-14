import mujoco
import numpy as np

class BarberEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("assets/so101_barber.xml")
        self.data = mujoco.MjData(self.model)
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()
    
    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        return self._get_obs(), 0.0, False, {}
    
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])
    
    def render(self):
        pass  # Rendering handled by viewer 