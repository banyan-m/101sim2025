# Robo-Barber

A reinforcement learning project for training a robotic barber using MuJoCo and Stable-Baselines3.

## Setup

1. Ensure you have Docker and VS Code with Remote Containers extension installed
2. Clone this repository
3. Open in VS Code and click "Reopen in Container" when prompted
4. The container will build and install all dependencies automatically

## Project Structure

- `assets/` - MuJoCo XML files for the robot and wig
- `env.py` - Gym-style BarberEnv implementation
- `train.py` - Stable-Baselines3 PPO training script
- `demo.py` - Script for recording demonstration videos

## Requirements

- Python 3.10
- MuJoCo 3.1.0
- Stable-Baselines3
- Gymnasium
- LeRobot (for SO-101 XML and helpers)

## License

This project uses MuJoCo under the free license. Make sure to have your MuJoCo license file in `~/.mujoco/`. 