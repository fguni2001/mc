from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use yolov8s.pt or yolov8m.pt for better accuracy

# Path to your video or folder of images
video_path = 'YOUR_INPUT_VIDEO.mp4'
output_dir = 'detections_yolo'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 1
detection_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)
    for det in results[0].boxes:
        # DeepSORT expects: [frame, id, x, y, w, h, score, class, visibility]
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        conf = float(det.conf[0])
        cls = int(det.cls[0])
        width, height = x2 - x1, y2 - y1
        detection_list.append([frame_id, -1, x1, y1, width, height, conf, cls, -1])

    frame_id += 1

cap.release()

# Save detections to a CSV/TXT file (like MOT format)
detection_file = os.path.join(output_dir, "YOUR_VIDEO.txt")
np.savetxt(detection_file, detection_list, fmt='%d,%d,%d,%d,%d,%d,%.4f,%d,%d')
print("Detections saved to", detection_file)
