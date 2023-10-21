import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect face
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # making rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Facial Detection', frame)

    # Running in the loop until Esc is pressed 
    key = cv2.waitKey(1)
    if key == 27:
        break
#closing the window
cap.release()
cv2.destroyAllWindows()
