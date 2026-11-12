from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

results = model("../images/img1.jpg")

print("these are result of an image ",results)

cv2.imshow("YOLO Detection", results[0].plot())

cv2.waitKey(0)
