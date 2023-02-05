import cv2
import numpy as np

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the face recognition model (OpenCV uses LBPH face recognizer by default)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-recognition.yml")

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop over the faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Predict the label of the face
        label, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if the confidence is less than 100 (100 is perfect match)
        if confidence < 100:
            # Allow access
            print("Access granted")
        else:
            # Deny access
            print("Access denied")

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Break the loop if the "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam
cap.release()

# Destroy all the windows
cv2.destroyAllWindows()
