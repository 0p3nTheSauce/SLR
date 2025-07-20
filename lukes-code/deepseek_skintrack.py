import cv2
import numpy as np
from utils


# Read image
frame = cv2.imread("face.jpg")

# Convert to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Define skin color range in HSV
lower_skin = np.array([0, 48, 80], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Threshold the HSV image to get skin
mask = cv2.inRange(hsv, lower_skin, upper_skin)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


cap = cv2.VideoCapture('./media/69241.mp4')
frames = []

while True:
  ret, frame = cap.read()
  if not ret:
    break
  
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv, lower_skin, upper_skin)
  
  # Find largest contour
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
  
  frames.append(frame)
  # cv2.imshow("Skin Tracking", frame)
  # if cv2.waitKey(1) & 0xFF == ord('q'):
  #     break

