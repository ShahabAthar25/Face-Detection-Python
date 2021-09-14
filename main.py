import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the frame
    cv.imshow('frame', frame)

    # Closing the window if q is pressed
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# releasing capture
cv.release()
cv.destryAllWindows()