import cv2

#choose cascade classifier in our case for fornt face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#reading the image source file and converting to gray image
gray_image = cv2.imread('/home/satish/PycharmProjects/fd/test.jpg')

# passing image file and and drawing ROI for detecting faces
faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(gray_image,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray_image[y:y+h, x:x+w]
    roi_color = gray_image[y:y+h, x:x+w]

#displaying the resulted image
cv2.imshow('img',gray_image)
cv2.waitKey(0)

#closes all windows we created
cv2.destroyAllWindows()