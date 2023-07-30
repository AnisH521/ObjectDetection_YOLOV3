import os
import cv2
import numpy as np

data_filename = "cat.jpeg"
weights_filename = "yolov3.weights"
config_filename = "yolov3.cfg"
classes_filename = "coco_classes.txt"

data_folder = "data/"
model_folder = "model/"
class_folder = "classes/"
output_folder = "output/"

data_path = os.path.join(data_folder, data_filename)
weights_path = os.path.join(model_folder, weights_filename)
config_path = os.path.join(model_folder, config_filename)
classes_path = os.path.join(class_folder, classes_filename)

image = cv2.imread(data_path)
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

try:
    with open(classes_path, 'r') as classes_file:
        class_names = classes_file.read().strip().split('\n')
except FileNotFoundError:
    print(f"Error: The {classes_filename} file was not found.")
    exit()

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

try:
    yolo_net = cv2.dnn.readNet(weights_path, config_path, "darknet")
except cv2.error:
    print("Error: One or more YOLO files were not found.")
    exit()

try:
    image = cv2.imread(data_path)
    
    Width = image.shape[1]
    Height = image.shape[0]
    
    scale = 0.00392
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    yolo_net.setInput(blob)

    def get_output_layers(net):
        layer_names = net.getLayerNames()   
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(class_names[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    outs = yolo_net.forward(get_output_layers(yolo_net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

    for i in range(len(boxes)):
        if i in indices:
            box = boxes[i]
            x, y, w, h = box
            draw_bounding_box(image, class_ids[i], confidences[i], x, y, x + w, y + h)

    # Display the image with detections
    cv2.imshow("Object Detection", image)

    # Wait for a key event and close the window when a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the output image to disk
    output_image_path = os.path.join(output_folder, "object-detection.jpg")
    cv2.imwrite(output_image_path, image)

    print(f"Object detection result saved as {output_image_path}")
except FileNotFoundError:
    print(f"Error: The image file '{image_filename}' was not found.")