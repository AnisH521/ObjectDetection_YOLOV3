# ObjectDetection_YOLOV3

This repository contains Example Python code for performing object detection using YOLOv3 (You Only Look Once version 3) on the COCO (Common Objects in Context) dataset. The YOLOv3 model is trained to detect 80 different object classes such as person, car, bicycle, dog, cat, etc.

### Features

- Utilizes the YOLOv3 pre-trained model with COCO weights for accurate and efficient object detection.
- Includes a "coco_classes.txt" file listing all the class names from the COCO dataset.
- Loads YOLO weights and configuration using OpenCV's `cv2.dnn.readNet` function.
- Draws bounding boxes and labels on the detected objects with confidence scores.
- Displays the image with detections and saves the result to"object-detection.jpg".

### Requirements

- Python 3.x
- OpenCV
- NumPy

### Usage

1. Download the YOLOv3 weights file from [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights).
2. Place the downloaded "yolov3.weights" file inside the "model" folder within the repository.
3. Run the Python script "object_detection.py" to detect objects in a sample image ("cat.jpeg").
4. The script will display the detected objects with bounding boxes and save the output as "object-detection.jpg".
