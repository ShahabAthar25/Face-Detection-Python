import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the frame

    # converting colors to gray
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # getting face cordinates
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # showing face cordiates
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv.imshow('Web Cam', frame)

    # Closing the window if q is pressed
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

# releasing capture
cap.release()
cv.destroyAllWindows()