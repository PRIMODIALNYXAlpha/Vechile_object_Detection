# Vechile_object_Detection

🚀 YOLO Object Detection & Line-Cross Counting with Ultralytics
This project implements object detection and tracking using Ultralytics YOLO, with a focus on counting objects that cross a predefined line in a video (e.g., vehicles crossing a virtual red line). It also supports GPU acceleration using CUDA to significantly speed up processing.
<img width="1597" height="881" alt="image" src="https://github.com/user-attachments/assets/875ca965-d5a9-4621-864d-71b7f5914d76" />

🧠 Algorithms Used
1. YOLO (You Only Look Once)
YOLO is a real-time object detection algorithm.

It uses a single neural network to divide the image into regions and predict bounding boxes and class probabilities for each region.

The Ultralytics implementation uses advanced versions (v8/NAS/RTDETR) for high accuracy.

2. Object Tracking
YOLO can perform object tracking via the .track() method, which uses a built-in BYTETracker to assign IDs to detected objects.

Each detected object has:

A unique track_id

A class label (e.g., car, person)

A bounding box

3. Line-Cross Counting
When the center (cx, cy) of a tracked object crosses a horizontal line (y = 430) and it hasn't crossed before:

We increment the class count.

We store the ID to prevent recounting.

⚙️ System Requirements
OS: Windows 10 or 11

Python: 3.10+

GPU: NVIDIA (e.g., GTX 1650 Ti)

CUDA: 12.8 (already installed with your driver)

RAM: 8GB or higher recommended

✅ Setup Instructions
1. 🛠️ Create and Activate a Virtual Environment
bash
Copy
Edit
cd Desktop/Object
python -m venv yolo
yolo\Scripts\activate
2. 📦 Install Required Packages (GPU Version)
bash
Copy
Edit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python
Replace cu118 with cu128 if PyTorch adds CUDA 12.8 support officially.

3. 🔍 Verify GPU is Available
In Python:

python
Copy
Edit
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should show "NVIDIA GeForce GTX 1650 Ti"
📘 Step-by-Step Working
1. Model Loading
python
Copy
Edit
from ultralytics import YOLO
model = YOLO('yolo11l.pt')
Loads a pretrained YOLO model. You can use:

yolov8n.pt – Nano (fastest, least accurate)

yolov8s.pt – Small

yolov8m.pt – Medium

yolov8l.pt – Large

yolov8x.pt – eXtreme (slowest, most accurate)

2. Open Video
python
Copy
Edit
cap = cv2.VideoCapture('test_video/4.mp4')
3. Detection + Tracking
python
Copy
Edit
results = model.track(frame, persist=True, classes=[1,2,3,5,6,7])
classes filters for relevant object types like vehicles and persons.

persist=True allows tracking across frames.

4. Draw Red Line
python
Copy
Edit
cv2.line(frame, (690, 430), (1130, 430), (0, 0, 255), 3)
5. Count Objects That Cross the Line
python
Copy
Edit
if cy > line_y_red and track_id not in crossed_ids:
    crossed_ids.add(track_id)
    class_counts[class_name] += 1
🖼️ Output
Bounding boxes and track IDs drawn on video frames

Class count displayed on the top-left

Red line represents the line to be crossed for counting

🧪 Sample Output
makefile
Copy
Edit
car: 12
truck: 3
bus: 1
person: 7
💡 Tips for Smooth Performance
Use .track() over .predict() to enable real-time tracking.

Run in GPU mode (torch.cuda.is_available() must be True).

Don’t overload the frame with too many boxes or unnecessary classes.

