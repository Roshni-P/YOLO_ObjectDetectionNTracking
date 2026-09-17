import cv2
import yoloDetector

# Initialize the detector
detector = yoloDetector.YOLOv8Detector(use_cuda=False)
if not detector.load_model(""):
    print("Error: Could not load YOLOv8 model.")
    exit()

# Open the MP4 video file
video_path = "PeopleStreetCloseView.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# 1. Create a resizable named window
window_name = "YOLOv8 Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# 2. Resize the display window to fit screen (e.g., 1280x720)
cv2.resizeWindow(window_name, 1280, 720)    

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video stream

    # Run detection on current frame
    detections = detector.detect(frame, conf_threshold=0.45, nms_threshold=0.5)

    # Draw results
    for det in detections:
        label = "Person" if det.class_id == 0 else "Vehicle"
        x, y, w, h = det.box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {det.confidence:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("YOLOv8 Detection", frame)
    
    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()