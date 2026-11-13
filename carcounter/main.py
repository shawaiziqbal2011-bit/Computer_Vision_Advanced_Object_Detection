import cv2
from ultralytics import YOLO
import cvzone
from sort import *
import numpy as np

cap = cv2.VideoCapture("cam_survelance.mp4")
model = YOLO("yolov8n.pt")

mask = cv2.imread("mask_cam_survelance.png", cv2.IMREAD_GRAYSCALE)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# x1,y1,x2,y2
limits = [200, 250, 630, 250]
totalCount =[]

while True:
    success, frame = cap.read()

    # video and mask heights and width is different trying to adjust it.
    resizedMask = cv2.resize(mask, (640, 360))
    # Performing bitwise and operation putting mask on the original frame
    imgRegion = cv2.bitwise_and(frame, frame, mask=resizedMask)
    results = model(imgRegion, stream=True)

    # Declare an empty array do save the detections
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Find the coordinates of the object that has been detected by the model
            # x1,y1,x2,y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            # Draw rectangle on the object that has been detected
            # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            bbox = (x1, y1, w, h)
            conf = int(box.conf[0] * 100)
            cls = int(box.cls[0])
            print("this is actual classname id ", cls)
            name = model.names[cls]
            # name = 'dummy'
            print("This is actual name", name)
            if ((name == "car" or name == "truck" or name == "bus"
                 or name == "motorbike") and conf > 0.3):
                cvzone.cornerRect(frame, bbox)
                cvzone.putTextRect(frame, f"Conf={conf} Name={name}", (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                   offset=3)

                # now save detections
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    # Draw the line
    cv2.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)

    for res in resultsTracker:
        x1, y1, x2, y2, ids = res
        # print("This is the unique id assigned to each vehicle ",ids)
        x1, y1, x2, y2, ids = int(x1), int(y1), int(x2), int(y2), int(ids)
        w, h = x2 - x1, y2 - y1
        bbox = (x1, y1, w, h)
        cvzone.cornerRect(frame, bbox)
        cvzone.putTextRect(frame, f"id={ids}", (max(0, x1), max(35, y1)), scale=1, thickness=1,
                           offset=3)
        #find the centerest point of the objects that are found and mark them with circle
        cx,cy = x1+w//2,y1+w//2
        cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
        if(limits[0]<cx<limits[2]and limits[1]-20<cy<limits[1]+20):
            #count function checks if this element exists in the array or not
            if(totalCount.count(ids)==0):
                totalCount.append(ids)
    cv2.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)


    cvzone.putTextRect(frame,f"Count:{len(totalCount)}",(50,50))

    # Original Frames
    cv2.imshow('frame', frame)
    # Show the original frame with mask
    # cv2.imshow('mask', imgRegion)
    cv2.waitKey(1)
