from env import BarberEnv
import numpy as np

def train():
    env = BarberEnv()
    
    print("Training barber robot...")
    
    for episode in range(10):
        obs = env.reset()
        print(f"Episode {episode + 1}")
        
        for step in range(100):
            # Random actions for now
            action = np.random.uniform(-1, 1, 4)  # 4 joints
            obs, reward, done, info = env.step(action)
            
            if step % 20 == 0:
                print(f"  Step {step}, obs shape: {obs.shape}")
    
    print("Training complete!")

if __name__ == "__main__":
    train() 