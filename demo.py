import mujoco
import mujoco.viewer
from env import BarberEnv
import numpy as np
import time
import argparse
from gymnasium.wrappers import RecordVideo

def demo(frames=400, record_video=False):
    print("Starting barbershop demo...")
    print("The robot will perform random movements.")
    
    if record_video:
        print(f"Recording {frames} frames to create MP4...")
        
        # Create environment with RGB rendering
        env = BarberEnv(render_mode="rgb_array")
        env = RecordVideo(env, "videos", episode_trigger=lambda x: True, video_length=frames)
        
        obs, _ = env.reset()
        for i in range(frames):
            # Smart barber movements
            action = np.array([
                0.3 * np.sin(i * 0.05),  # Base rotation
                0.5,                      # Shoulder up
                -0.8,                     # Elbow bend
                0.2 * np.cos(i * 0.03)   # Wrist adjust
            ])
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if info.get('hair_cut', False):
                print(f"Frame {i}: Hair cut! Reward: {reward}")
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        env.close()
        print("Video saved to videos/ directory")
        
    else:
        # Interactive viewer (fallback to headless if no display)
        env = BarberEnv()
        
        try:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                for i in range(frames):
                    # Random barber movements
                    action = 0.5 * np.sin(i * 0.01) * np.ones(4)
                    env.step(action)
                    
                    viewer.sync()
                    time.sleep(0.01)
        except Exception as e:
            print(f"Interactive viewer failed: {e}")
            print("Running headless simulation instead...")
            
            obs, _ = env.reset()
            for i in range(frames):
                action = 0.5 * np.sin(i * 0.01) * np.ones(4)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if i % 20 == 0:
                    print(f"Step {i}: reward={reward:.3f}, hair_cut={info.get('hair_cut', False)}")
                
                if terminated or truncated:
                    obs, _ = env.reset()
    
    print("Demo complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=400, help="Number of frames to run")
    parser.add_argument("--record", action="store_true", help="Record video instead of interactive viewer")
    args = parser.parse_args()
    
    demo(args.frames, args.record) 