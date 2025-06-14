import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BarberEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    
    def __init__(self):
        super(BarberEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        
    def step(self, action):
        # Execute one time step within the environment
        pass
        
    def reset(self, seed=None):
        # Reset the state of the environment to an initial state
        pass
        
    def render(self):
        # Render the environment to the screen
        pass 