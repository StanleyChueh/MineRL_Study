import gym
import json
import time
import numpy as np
from custom_env import CustomTreeCuttingEnv  # Import your custom environment

# Path to the JSONL file and Minecraft environment setup
JSONL_FILE = "mc-0.jsonl"
ENV_NAME = "CustomTreeCutting-v0"

# Initialize the custom environment
env = gym.make(ENV_NAME)
env.seed(2143)  # Set the seed for reproducibility
obs = env.reset()

# Load JSONL file
try:
    with open(JSONL_FILE, "r") as f:
        action_log = [json.loads(line) for line in f]
except Exception as e:
    print(f"Error loading JSONL file: {e}")
    exit(1)

done = False
cumulative_reward = 0.0
step_limit = 10000  # Limit the steps to avoid infinite loops

try:
    for step_idx, action in enumerate(action_log):
        if done or step_idx >= step_limit:
            print(f"Stopping replay at step {step_idx}. Done: {done}")
            break

        # Parse the JSONL data
        action_data = action.get("action", {})
        mouse_data = action_data.get("mouse", {})
        keyboard_data = action_data.get("keyboard", {})

        # Extract camera and button data
        camera_action = mouse_data.get("camera", [0.0, 0.0])
        mouse_buttons = mouse_data.get("buttons", [])
        keyboard_keys = keyboard_data.keys()

        # Convert JSONL action into Minecraft-compatible action
        minecraft_action = {
            "ESC": keyboard_data.get("ESC", 0),
            "camera": camera_action,
            "forward": keyboard_data.get("forward", 0),
            "back": keyboard_data.get("back", 0),
            "left": keyboard_data.get("left", 0),
            "right": keyboard_data.get("right", 0),
            "jump": keyboard_data.get("jump", 0),
            "sprint": keyboard_data.get("sprint", 0),
            "sneak": keyboard_data.get("sneak", 0),
            "attack": 1 if 0 in mouse_buttons else 0,
            "use": 1 if 2 in mouse_buttons else 0,
            "hotbar.1": keyboard_data.get("hotbar.1", 0),
            "hotbar.2": keyboard_data.get("hotbar.2", 0),
            "hotbar.3": keyboard_data.get("hotbar.3", 0),
            "hotbar.4": keyboard_data.get("hotbar.4", 0),
            "hotbar.5": keyboard_data.get("hotbar.5", 0),
            "hotbar.6": keyboard_data.get("hotbar.6", 0),
            "hotbar.7": keyboard_data.get("hotbar.7", 0),
            "hotbar.8": keyboard_data.get("hotbar.8", 0),
            "hotbar.9": keyboard_data.get("hotbar.9", 0),
        }

        # Debugging: Log parsed actions
        print(f"Step {step_idx + 1}: Parsed Action={minecraft_action}")
        print(f"Step {step_idx + 1}: Camera Action={minecraft_action['camera']}")

        # Apply the action to the environment
        try:
            obs, reward, done, info = env.step(minecraft_action)
        except Exception as e:
            print(f"Error during env.step at step {step_idx}: {e}")
            break

        # Accumulate reward
        cumulative_reward += reward

        # Debugging: Print environment response
        print(f"Step {step_idx + 1}: Observation={'Keys Available' if isinstance(obs, dict) else 'No Keys'}")
        print(f"Step {step_idx + 1}: Reward={reward}, Cumulative Reward={cumulative_reward}, Done={done}, Info={info}")

        # Add a delay to match the recorded tick rate (default 50ms per tick)
        tick_duration = action.get("timestamp", 50.0) / 1000.0
        time.sleep(min(tick_duration, 0.1))  # Cap sleep time to avoid excessive delays

        # Optionally render the environment (if supported by the environment)
        try:
            env.render()
        except Exception as e:
            print(f"Render error at step {step_idx}: {e}")
            break

except KeyboardInterrupt:
    print("Replay interrupted by user.")
finally:
    env.close()
    print(f"Replay complete. Total Reward: {cumulative_reward}")
