import mujoco
import mujoco.viewer
from env import BarberEnv
import numpy as np
import time

def demo():
    env = BarberEnv()
    
    print("Starting barbershop demo...")
    print("The robot will perform random movements.")
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for i in range(1000):
            # Random barber movements
            action = 0.5 * np.sin(i * 0.01) * np.ones(4)
            env.step(action)
            
            viewer.sync()
            time.sleep(0.01)
    
    print("Demo complete!")

if __name__ == "__main__":
    demo() 