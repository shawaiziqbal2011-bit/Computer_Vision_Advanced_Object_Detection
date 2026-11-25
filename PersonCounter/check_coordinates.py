
import cv2

cap = cv2.VideoCapture("escalator_survelance.mp4")

success, frame = cap.read()

def mousePoints(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"X: {x}, Y: {y}")
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Coordinates", frame)

cv2.imshow("Select Coordinates", frame)
cv2.setMouseCallback("Select Coordinates", mousePoints)

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
