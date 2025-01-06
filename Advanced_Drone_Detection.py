import cv2
import torch
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/vackost/Advanced-Aerial-Drone-Detection-System1/best.pt', source='github')

# Set video source (webcam or video file)
cap = cv2.VideoCapture(0)

# Define the classes you want to detect
classes = ['Drone']

# Initialize rectangle coordinates
rectangle_coords = [(50, 50), (250, 50), (250, 250), (50, 250)]
rectangle_drag = False
drag_corner = -1
tracked_objects = []

# Mouse event handler
def mouse_event(event, x, y, flags, param):
    global rectangle_coords, rectangle_drag, drag_corner
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, corner in enumerate(rectangle_coords):
            if abs(corner[0] - x) <= 10 and abs(corner[1] - y) <= 10:
                rectangle_drag = True
                drag_corner = i
                break
    elif event == cv2.EVENT_LBUTTONUP:
        rectangle_drag = False
    elif event == cv2.EVENT_MOUSEMOVE and rectangle_drag:
        rectangle_coords[drag_corner] = (x, y)

# Set mouse callback
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouse_event)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Convert frame to YOLOv5 format
    img = Image.fromarray(frame[..., ::-1])
    results = model(img, size=640)

    # Process detection results
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result.tolist()
        if conf > 0.5 and classes[int(cls)] in classes:
            # Calculate bounding box dimensions
            bbox = (int(x1), int(y1), int(x2), int(y2))
            width = int(x2 - x1)
            height = int(y2 - y1)

            # Always draw the rectangle for tracking
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            text_conf = "{:.2f}%".format(conf * 100)
            text_size = "W: {}, H: {}".format(width, height)
            cv2.putText(frame, text_conf, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, text_size, (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Add the object to tracking list
            if bbox not in tracked_objects:
                tracked_objects.append(bbox)

            # Check if the object intersects with the restricted area
            if width >= 150 and height >= 150:
                if rectangle_coords[0] != rectangle_coords[1]:
                    if any(rectangle_coords[0][0] <= x <= rectangle_coords[2][0] and rectangle_coords[0][1] <= y <= rectangle_coords[2][1]
                           for x, y in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]):
                        cv2.putText(frame, "Warning: Drone Detected Under Restricted Area!",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw the adjustable rectangle
    for i in range(4):
        cv2.circle(frame, rectangle_coords[i], 5, (0, 255, 0), -1)
        cv2.line(frame, rectangle_coords[i], rectangle_coords[(i + 1) % 4], (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
