import cv2
import sys

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #converting to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detects objects of different sizes in the input image
    #objects are returned as a list of rectangles
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    #press 'c' to close the live camera window
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
