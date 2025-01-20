import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import non_max_suppression, scale_boxes

# Load YOLOv5 model
weights = 'epoch300.pt'
device = select_device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights, device=device)

# Load image
image_path = 'minecraft_tree.png'
img0 = cv2.imread(image_path)
assert img0 is not None, "Image not found."

# Display the input image
plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
plt.title("Input Image")
plt.axis("off")
plt.show()

# Prepare image for inference
img = cv2.resize(img0, (640, 640))
img = img[..., ::-1]  # Convert BGR to RGB
img = img / 255.0
img = np.expand_dims(img, axis=0).transpose(0, 3, 1, 2)
img = torch.from_numpy(img).float().to(device)

# Run inference
pred = model(img)
print("Raw model output:", pred)

# Apply Non-Max Suppression
pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.3)
print("Post-NMS predictions:", pred)

# Process detections and visualize results
if pred[0] is not None and len(pred[0]) > 0:
    # Rescale boxes to original image size
    pred_boxes = pred[0].cpu().numpy()
    pred_boxes[:, :4] = scale_boxes(img.shape[2:], pred_boxes[:, :4], img0.shape).round()

    # Draw bounding boxes on the image
    for box in pred_boxes:
        x1, y1, x2, y2, conf, cls = box
        cv2.rectangle(img0, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"Tree {conf:.2f}"
        cv2.putText(img0, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save and display results
    cv2.imwrite("detection_debug.jpeg", img0)
    print("Detection saved to 'detection_debug.jpeg'")

    img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Detection Results")
    plt.axis("off")
    plt.show()
else:
    print("No detections found.")
