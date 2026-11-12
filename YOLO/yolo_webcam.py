import cv2
from ultralytics import YOLO
import cvzone

cap  = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
model = YOLO("yolov8n.pt")

while True :
    success , frame = cap.read()
    results = model(frame,stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Find the coordinates of the object that has been detected by the model
            # x1,y1,x2,y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            # Draw rectangle on the object that has been detected
            # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            bbox = (x1, y1, w, h)
            conf = int(box.conf[0]*100)
            cls = int(box.cls[0]*100)
            name = model.names[cls]

            cvzone.cornerRect(frame, bbox)
            cvzone.putTextRect(frame,f"Conf={conf} Name={name}",(max(0,x1),max(35,y1)),scale=1)

            # print(x1, y1, x2, y2)

    cv2.imshow('frame',frame)
    cv2.waitKey(1)