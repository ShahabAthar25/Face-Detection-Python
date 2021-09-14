import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # converting colors to gray
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the frame
    cv.imshow('frame', frame)

    # getting face cordinates
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # showing face cordiates
    for (x, y, w, h) in faces:
        print(x, y, w, h)

    # Closing the window if q is pressed
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# releasing capture
cv.release()
cv.destryAllWindows()