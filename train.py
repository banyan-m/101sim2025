from env import BarberEnv
import numpy as np
from stable_baselines3 import PPO
import argparse

def train(timesteps=2000):
    env = BarberEnv()
    
    print(f"Training barber robot with PPO for {timesteps} timesteps...")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Initialize PPO agent
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
    
    # Train the agent
    model.learn(total_timesteps=timesteps)
    
    print("Training complete!")
    
    # Test the trained agent
    obs, _ = env.reset()
    for i in range(10):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, hair_cut={info.get('hair_cut', False)}")
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2000, help="Number of training timesteps")
    args = parser.parse_args()
    
    train(args.timesteps) 