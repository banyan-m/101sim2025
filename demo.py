import gymnasium as gym
from stable_baselines3 import PPO
from env import BarberEnv

def record_demo(model_path, output_path, num_episodes=1):
    # Load the trained agent
    model = PPO.load(model_path)
    env = BarberEnv()
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _, _ = env.step(action)
            env.render()

if __name__ == "__main__":
    record_demo("barber_ppo", "demo.mp4") 