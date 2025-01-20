import gym
import pygame
import numpy as np
import os
import time
import json
from custom_env import CustomTreeCuttingEnv  # Import the custom environment
import torch  # For loading the .pt file
from torchvision import transforms
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
import cv2

# Utility functions
def preprocess_image(image, preprocess):
    """Preprocess the image for the YOLO model."""
    return preprocess(image).unsqueeze(0)


def render_text(screen, text, position, font_size=24, color=(255, 255, 255)):
    """Renders text on the screen."""
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

# Load your tree detection model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reinitialize the model architecture
# Load the YOLOv5 model using DetectMultiBackend
tree_detection_model = DetectMultiBackend(weights="epoch300.pt", device=device)  # Load weights
tree_detection_model.eval()  

# Preprocessing for the model
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert to Tensor and normalize to [0, 1]
    transforms.Resize((640, 640))  # Resize to YOLOv5 input size
])

# Set up directories for video and logs
samples = 1
output_video_path = os.getcwd()
video_dir = f"{output_video_path}/data/labeller-training/video"
os.makedirs(video_dir, exist_ok=True)

for i in range(samples):
    OUTPUT_VIDEO_FILE = f"{video_dir}/mc-{i}.mp4"
    ACTION_LOG_FILE = f"{video_dir}/mc-{i}.jsonl"

    # Initialize pygame
    pygame.init()
    FPS = 30
    RESOLUTION = (640, 360)  # Resolution at which to capture and save the video
    screen = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption('Minecraft')
    SENS = 0.05

    # Initialize the OpenCV video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, FPS, RESOLUTION)

    pygame.mouse.set_visible(False)
    pygame.mouse.set_pos(screen.get_width() // 2, screen.get_height() // 2)  # Center the mouse
    pygame.event.set_grab(True)
    prev_mouse_x, prev_mouse_y = screen.get_width() // 2, screen.get_height() // 2

    # Define key mappings for actions
    key_to_action_mapping = {
        pygame.K_w: "forward",
        pygame.K_s: "back",
        pygame.K_a: "left",
        pygame.K_d: "right",
        pygame.K_SPACE: "jump",
        pygame.K_LSHIFT: "sprint",
        pygame.K_LCTRL: "sneak",
        pygame.K_1: "hotbar.1",
        pygame.K_2: "hotbar.2",
        pygame.K_3: "hotbar.3",
        pygame.K_4: "hotbar.4",
        pygame.K_5: "hotbar.5",
        pygame.K_6: "hotbar.6",
        pygame.K_7: "hotbar.7",
        pygame.K_8: "hotbar.8",
        pygame.K_9: "hotbar.9",
        pygame.K_e: "inventory",
        pygame.K_f: "swapHands",
        pygame.K_g: "drop",
        pygame.K_t: "pickItem",
        pygame.K_ESCAPE: "ESC",
        pygame.K_q: "quit",
        pygame.K_r: "reload",
    }

    mouse_to_action_mapping = {
        0: "attack",  # Left mouse button
        2: "use",     # Right mouse button
    }

    action_log = []

    # Initialize the custom environment
    env = CustomTreeCuttingEnv()  # Using your custom environment
    obs = env.reset()
    cumulative_reward = 0
    reward = 0.0
    step_count = 0
    hit_tree = False

    done = False
    try:
        while not done:
            current_time = int(time.time() * 1000)

            # Display the observation in pygame
            pov_image = obs['pov']
            if pov_image is None:
                print("Warning: pov_image is None.")
                continue

            # Resize and convert the image to match VideoWriter requirements
            pov_image = cv2.resize(pov_image, RESOLUTION)  # Ensure resolution matches VideoWriter
            pov_image = cv2.cvtColor(pov_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

            # Write the frame to the video
            out.write(pov_image)

            pov_image = pov_image.copy() 
            pov_image_for_model = preprocess(pov_image).unsqueeze(0).to(device) # Preprocess for the model

            tree_detection_output = tree_detection_model(pov_image_for_model)

            # Apply Non-Max Suppression (NMS)
            detections = non_max_suppression(tree_detection_output, conf_thres=0.25, iou_thres=0.45)


            # Check if any detection is a tree
            if detections[0] is not None and len(detections[0]) > 0:
                # Extract classes from detections
                detected_classes = detections[0][:, -1].cpu().numpy()  # Extract class IDs
                tree_detected = any(cls in [0, 1] for cls in detected_classes)  # Check for tree class (adjust IDs as needed)
            else:
                tree_detected = False

            pov_image = np.flip(pov_image, axis=1)  # Flip horizontally for correct orientation
            pov_image = np.rot90(pov_image)  # Rotate for pygame rendering
            # Display the environment
            pov_surface = pygame.surfarray.make_surface(pov_image)
            screen.blit(pov_surface, (0, 0))

            # Display text information
            render_text(screen, f"Step: {step_count}", (10, 10), font_size=24, color=(255, 255, 255))
            render_text(screen, f"Reward: {reward:.2f}", (10, 40), font_size=24, color=(255, 255, 255))
            render_text(screen, f"Cumulative Reward: {cumulative_reward:.2f}", (10, 70), font_size=24, color=(255, 255, 255))
            render_text(screen, f"Tree Detected: {'Yes' if tree_detected else 'No'}", (10, 100), font_size=24, color=(0, 255, 0) if tree_detected else (255, 0, 0))
            render_text(screen, f"Hit Tree: {'Yes' if hit_tree else 'No'}", (10, 130), font_size=24, color=(0, 255, 0) if hit_tree else (255, 0, 0))

            # Update the display
            pygame.display.flip()

            # Capture keyboard and mouse inputs
            action = {key: 0 for key in env.action_space.spaces.keys()} 
            keys = pygame.key.get_pressed()
            for key, action_name in key_to_action_mapping.items():
                if keys[key]:
                    action[action_name] = 1

            mouse_buttons = pygame.mouse.get_pressed()
            for idx, pressed in enumerate(mouse_buttons):
                if pressed:
                    button_action = mouse_to_action_mapping.get(idx)
                    if button_action:
                        action[button_action] = 1

            # Handle camera movement
            camera_delta_x = 0  # Initialize camera deltas
            camera_delta_y = 0

            for event in pygame.event.get():
                if event.type == pygame.MOUSEMOTION:
                    # Capture mouse movement without resetting the position
                    camera_delta_x += event.rel[0] * SENS  # Horizontal movement
                    camera_delta_y += event.rel[1] * SENS  # Vertical movement

            # Log the camera movement
            action["camera"] = [camera_delta_y, camera_delta_x]

            pygame.mouse.set_pos(screen.get_width() // 2, screen.get_height() // 2)
            prev_mouse_x, prev_mouse_y = screen.get_width() // 2, screen.get_height() // 2

            # Apply the action in the environment
            obs, reward, done, info = env.step(action)
            cumulative_reward += reward
            step_count += 1

            # Determine if a tree was hit
            hit_tree = action.get("attack", 0) == 1 and tree_detected and env._is_near_tree(obs)

            # Log the action and observation
            action_log.append({
                "action": {
                    "keyboard": {key: value for key, value in action.items() if key in key_to_action_mapping.values()},
                    "mouse": {
                        "camera": action.get("camera", [0.0, 0.0]),  # Default: no mouse movement
                        "buttons": [idx for idx in mouse_to_action_mapping if action.get(mouse_to_action_mapping[idx], 0)],
                    }
                },
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "tree_detected": tree_detected,  # Check if a tree is detected
                "hit_tree": hit_tree,
                "timestamp": current_time
            })

            # Print step info
            print(f"Step {step_count}: Reward={reward}, Cumulative Reward={cumulative_reward}, Tree Detected={tree_detected}")

            # Check for exit condition
            for event in pygame.event.get():
                if event.type == pygame.QUIT or keys[pygame.K_q]:
                    done = True

    except KeyboardInterrupt:
        print("Recording interrupted.")
    finally:
        # Save the action log
        with open(ACTION_LOG_FILE, "w") as f:
            for entry in action_log:
                f.write(json.dumps(entry) + "\n")

        out.release()
        pygame.quit()
        env.close()

        print(f"Video saved to {OUTPUT_VIDEO_FILE}")
        print(f"Action log saved to {ACTION_LOG_FILE}")
