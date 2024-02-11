import os
from ultralytics import YOLO
import cv2


model_path = os.path.join('runs-20240210T195922Z-001', 'runs', 'detect', 'train', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

# Specify the image path
image_path = "youtube-537_jpg.rf.1ebb382647700ce009f134ae40bdd654.jpg"  # Replace with your image path

# Read the image
image = cv2.imread(image_path)

# Detect objects in the image
results = model(image)[0]

# Loop through detections and draw bounding boxes and labels
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Display the image with detections
cv2.imshow("Agriculture detection", image)
cv2.waitKey(0)  # Wait for any key press before closing the window
cv2.destroyAllWindows()
