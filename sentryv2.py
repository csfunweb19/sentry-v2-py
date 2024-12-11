
import cv2
import time
import numpy as np
import os
# Define file paths
yolo_weights = r'C:\Users\quesa\Projects\sentryv2\yolov3.weights'
yolo_cfg = r'C:\Users\quesa\Projects\sentryv2\yolov3.cfg'
coco_names = r'C:\Users\quesa\Projects\sentryv2\coco.names'

# Check file existence
for file_path in [yolo_weights, yolo_cfg, coco_names]:
    if os.path.exists(file_path):
        print(f"Found: {file_path}")
    else:
        print(f"Missing: {file_path}")


if not os.path.exists(yolo_weights) or not os.path.exists(yolo_cfg) or not os.path.exists(coco_names):
    print("Error: One or more YOLO files are missing.")
    exit()


#YOLO
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)  # Path to the YOLO model weights and config
with open(coco_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]


layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]


# (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

frame_rate = 30
time_between_frames = 1 / frame_rate  

# Start recording
start_time = time.time()

while True:
    # Capture the frame-by-frame
    ret, frame = cap.read()

    # If the frame is captured successfully, display it
    if ret:
        # Display the frame in a window
        cv2.imshow('Webcam Feed', frame)
        height, width, channels = frame.shape
        #cv2.dnn.blobFromImage(image, scale_factor, size, mean, swapRB, crop)

        blob = cv2.dnn.blobFromImage(frame, float(1/255), (416, 416), (0,0,0), True, crop=False)
        net.setInput(blob)

        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and class_id == 0:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1]* height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x-w/2)
                    y = int(center_y-h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = f"Person {confidences[i]: .2f}"
                color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show frames/per/secong on the screen'/
        # fps = int(1 / time_between_frames)
        fps = 60

        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam Feed', frame)

        # Display the frame at the set frame rate
        if cv2.waitKey(1) & 0xFF == ord('e'):  # Press 'e' to stop
            print("Recording stopped.")
            break

    # Wait for the next frame to maintain the 30 FPS rate
    time.sleep(time_between_frames)
   

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()


