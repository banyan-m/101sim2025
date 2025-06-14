from stable_baselines3 import PPO
from env import BarberEnv

def main():
    # Create the environment
    env = BarberEnv()
    
    # Initialize the agent
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the agent
    model.learn(total_timesteps=1000000)
    
    # Save the agent
    model.save("barber_ppo")

if __name__ == "__main__":
    main() 