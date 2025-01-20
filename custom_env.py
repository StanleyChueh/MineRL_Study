import gym
import minerl
import numpy as np
import cv2
import torch
from torchvision import transforms
from yolov5.utils.general import non_max_suppression


class CustomTreeCuttingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Load the MineRL environment
        self.env = gym.make('MineRLBasaltFindCave-v0')
        self.env.seed(2143)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # Define custom environment-specific variables
        self.tree_cutting_reward = 10  # Reward for cutting a tree
        self.proximity_reward = 2      # Reward for approaching trees
        self.penalty_for_unnecessary_action = -1
        self.done = False

        # Add a task attribute to match MineRL expectations
        self.task = self.env.task  # Use the same task from the base environment

        # Load the YOLOv5 tree detection model
        self.model = self._load_yolov5_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        # Define the preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((640, 640)),  # Adjust size as needed for YOLOv5
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image
        ])

    def _load_yolov5_model(self):
        """Load the YOLOv5 model."""
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='epoch300.pt', force_reload=True)
        return model

    def reset(self):
        """Resets the environment."""
        self.consecutive_cuts = 0  # Reset consecutive cut counter
        self.done = False
        obs = self.env.reset()  # Reset the underlying environment
        return obs  # Return the observation

    def step(self, action):
        """Steps through the environment with custom rewards."""
        obs, reward, done, info = self.env.step(action)

        # Initialize custom reward
        custom_reward = 0

        # Reward for cutting trees
        if self._is_tree_cut(action, obs):
            custom_reward += self.tree_cutting_reward
            print(f"Tree cut! Reward added: {self.tree_cutting_reward}")
            if getattr(self, "consecutive_cuts", 0) >= 1:  # Bonus for consecutive cuts
                custom_reward += 2  # Additional bonus
                print("Bonus reward for consecutive cuts: +2")
            self.consecutive_cuts = getattr(self, "consecutive_cuts", 0) + 1
        else:
            self.consecutive_cuts = 0  # Reset consecutive cuts if the agent stops cutting

        # Dynamic reward for moving closer to trees
        distance_to_tree = self._distance_to_nearest_tree(obs)
        if distance_to_tree != float('inf'):  # Ensure valid distance
            # Proximity reward increases as the agent gets closer to the tree
            proximity_reward = self.proximity_reward / (distance_to_tree + 1)
            custom_reward += proximity_reward
            print(f"Proximity reward: {proximity_reward}, Distance to tree: {distance_to_tree}")

        # Penalize unnecessary actions
        if self._is_unnecessary_action(action):
            custom_reward += self.penalty_for_unnecessary_action
            print(f"Unnecessary action penalty applied: {self.penalty_for_unnecessary_action}")

        # Combine custom rewards with the base reward
        reward += custom_reward
        print(f"Step Reward: {reward}, Custom Reward: {custom_reward}, Cumulative Reward: {reward + custom_reward}")

        return obs, reward, done, info



    def render(self, mode='human'):
        """Renders the environment."""
        self.env.render(mode=mode)

    def close(self):
        """Cleans up the environment."""
        self.env.close()

    def _is_tree_cut(self, action, obs):
        is_attack = action.get("attack", 0) == 1
        is_near = self._is_near_tree(obs)
        print(f"Tree cut condition: is_attack={is_attack}, is_near={is_near}")
        return is_attack and is_near

    def _is_near_tree(self, obs):
        """Checks if the agent is near a tree."""
        if obs is None:
            print("Warning: 'obs' is None in _is_near_tree.")
            return False

        # Use the YOLOv5 model to identify tree positions
        tree_positions = self._find_trees(obs)

        if not tree_positions:
            print("No trees detected.")
            return False

        # Get the agent's position as a NumPy array (x, y, z)
        agent_pos = np.array([
            obs.get("xpos", 0),
            obs.get("ypos", 0),
            obs.get("zpos", 0)
        ])

        # Initialize variables to calculate proximity reward
        closest_distance = float('inf')

        for tree_pos in tree_positions:
            try:
                # Convert tree position to NumPy array
                tree_pos_np = np.array([
                    element.cpu().numpy() if isinstance(element, torch.Tensor) else element
                    for element in tree_pos  # Use x, y, z
                ])
                # Calculate the Euclidean distance
                distance = np.linalg.norm(agent_pos - tree_pos_np)
                closest_distance = min(closest_distance, distance)  # Track the closest tree

            except Exception as e:
                print(f"Error processing tree position {tree_pos}: {e}")

        if closest_distance < float('inf'):
            print(f"Closest tree distance: {closest_distance}")
            return closest_distance  # Return the distance instead of a fixed "True/False"

        return False


    
    def _find_trees(self, obs):
        """Find tree positions using YOLOv5 detection."""
        if obs is None:
            print("Warning: 'obs' is None in _find_trees.")
            return []

        # Extract the POV image
        pov_image = obs.get('pov', None)
        if pov_image is None:
            print("Warning: 'pov' not found in obs.")
            return []

        # Ensure the image is contiguous in memory
        pov_image = pov_image.copy()

        # Preprocess the image and perform inference
        pov_image_for_model = self.preprocess(pov_image).unsqueeze(0).to(self.device)

        # Perform detection
        with torch.no_grad():
            tree_detection_output = self.model(pov_image_for_model)

        # Apply Non-Max Suppression (NMS)
        detections = non_max_suppression(tree_detection_output, conf_thres=0.25, iou_thres=0.45)

        # Parse detections to extract tree positions
        tree_positions = []
        if detections[0] is not None and len(detections[0]) > 0:
            for det in detections[0]:
                try:
                    # Extract and scale bounding box center coordinates
                    x_center = det[0].item()  # X-center of the bounding box
                    y_center = det[1].item()  # Y-center of the bounding box
                    class_id = int(det[5].item())  # Class ID as integer

                    if class_id in [0, 1]:  # Adjust tree class IDs as needed
                        # Append position (x, y, z)
                        tree_positions.append((x_center, y_center, 0))  # Z is hardcoded as 0
                except Exception as e:
                    print(f"Error processing detection: {e}")

        # Debugging log
        if tree_positions:
            print(f"Tree positions: {tree_positions}")
        else:
            print("No trees detected.")

        return tree_positions


    def _is_unnecessary_action(self, action):
        """Checks if an action is unnecessary."""
        if action.get("jump", 0) == 1:
            # Only check for tree proximity if 'obs' is not None
            if not hasattr(self, "current_obs") or self.current_obs is None:
                print("Warning: 'current_obs' is not set, skipping _is_near_tree.")
                return True
            if not self._is_near_tree(self.current_obs):
                return True
        return False


    def _distance_to_nearest_tree(self, obs):
        """Calculate the distance to the nearest tree."""
        tree_positions = self._find_trees(obs)

        # Get the agent's position as a NumPy array
        agent_pos = np.array([obs.get("xpos", 0), obs.get("ypos", 0), obs.get("zpos", 0)])

        # Calculate distances to all detected trees
        distances = []
        for tree_pos in tree_positions:
            # Convert tree position to CPU and NumPy
            tree_pos_np = np.array([
                element.cpu().numpy() if isinstance(element, torch.Tensor) else element
                for element in tree_pos[:2]  # Use only x, y positions
            ])
            distance = np.linalg.norm(agent_pos[:2] - tree_pos_np)
            distances.append(distance)

        # Return the minimum distance or infinity if no trees are detected
        return min(distances) if distances else float('inf')



# Register the custom environment
from gym.envs.registration import register

register(
    id='CustomTreeCutting-v0',  # Unique identifier
    entry_point='custom_env:CustomTreeCuttingEnv',  # Path to the custom class
)
