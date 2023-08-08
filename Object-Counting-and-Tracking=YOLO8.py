# Import required libraries
from ultralytics import YOLO
import cv2
import math
from sort import *

# Open the input video
cap = cv2.VideoCapture('cars.mp4')

# Load YOLO model
model = YOLO('yolov8m.pt')

# Load the region detection mask
mask = cv2.imread('images/mask1.png')

# List of class names in YOLO model
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]  # List of class names

# Get the frames per second (fps) of the input video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the output video format and parameters
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize a variable to check if the VideoWriter has been successfully opened
is_video_writer_initialized = False


# Create a tracker for object tracking
tracker = Sort(max_age=20)

# Detection limits for vehicles
limits = [400, 297, 673, 297]
totalCount = []
vehicleTrackDict = {}  # Dictionary for tracking and counting vehicles

while True:
    success, img = cap.read()
    imagRegion = cv2.bitwise_and(img, mask)

    # Perform object detection using YOLO model
    results = model(imagRegion, stream=True)
    detections = np.empty((0, 5))

    # Process detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Detect desired objects within confidence range
            if currentClass in ['car', 'bus', 'truck', 'motorbike'] and conf > 0.3:
                color = (0, 255, 255) if currentClass == 'car' else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update object tracking
    resultsTracker = tracker.update(detections)

    # Analyze tracking results
    for result in resultsTracker:
        x1, y1, x2, y2, _id = result
        x1, y1, x2, y2, _id = int(x1), int(y1), int(x2), int(y2), int(_id)

        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        image = cv2.putText(img, f'{_id}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if _id not in totalCount:
                totalCount.append(_id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
            if _id not in vehicleTrackDict:
                vehicleTrackDict[_id] = currentClass

    # Display total object count
    cv2.putText(img, f' Total All Object: {len(totalCount)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 255), 6)

    # Display object count for each category
    for vehicle_type in ['car', 'bus', 'truck', 'motorbike']:
        count = sum(1 for v_id, v_type in vehicleTrackDict.items() if v_type == vehicle_type)
        # Display object count for each category on the image
        cv2.putText(img, f' {vehicle_type}: {count}', (800, 10 + 40 * (classNames.index(vehicle_type) + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (139, 195, 75), 4)
        currenClass_found = True  # For video saving

    if not is_video_writer_initialized:
        # Initialize the VideoWriter only once
        out = cv2.VideoWriter('output_conter.mp4', fourcc, fps, (frame_width, frame_height), True)
        is_video_writer_initialized = True

    if currenClass_found:
        out.write(img)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

# Release resources and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
