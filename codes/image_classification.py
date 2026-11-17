import os
import cv2
from ultralytics import YOLO
model_path = os.path.join(os.getcwd(), 'best.pt')

# Load the YOLO model
model = YOLO(model_path)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting live object detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO prediction
    results = model(frame)[0]

    label_to_show = ""
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = box.conf[0].item()
        label = f"{model.names[cls]} {conf:.2f}"
        label_to_show = model.names[cls]  # Update label to show at top-left

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw top-left label with black background
    if label_to_show:
        text_size, _ = cv2.getTextSize(label_to_show, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        cv2.rectangle(frame, (5, 5), (5 + text_w + 10, 5 + text_h + 10), (0, 0, 0), -1)
        cv2.putText(frame, label_to_show, (10, 10 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display result
    cv2.imshow("YOLOv8 Live Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
