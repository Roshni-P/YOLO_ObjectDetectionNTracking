import cv2
import yoloWithCpp

detector = yoloWithCpp.ObjectDetector(use_cuda=False)
if not detector.load_model(""):
    print("Error: Could not load YOLOv8 model.")
    exit()

yoloWithCpp.execute()
