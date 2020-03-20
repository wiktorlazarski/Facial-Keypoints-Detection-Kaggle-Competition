import tensorflow as tf
import cv2 as cv
import numpy as np

def preprocess_frame(frame):
    '''Method preprocessing frame so that it can be feed to a neural network.

        Parameter:
        frame (ndarray): frame to be processed.

        Returns:
        preprocessed (ndarray): preprocessed frame.
    '''
    preprocessed = cv.resize(frame, (96, 96))
    preprocessed = preprocessed.reshape((1, 96, 96, 1))
    preprocessed = np.asarray(preprocessed, dtype=np.float32) / 255
    return preprocessed

# Load models
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model("facial_keypoints_detection.h5")

# Camera capture loop
cap = cv.VideoCapture(0)
while cap.isOpened(): 

    # Capture & display original frame
    ret, frame = cap.read()
    cv.imshow('Original', frame)

    # Extract first found face from a frame
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    # Continue if no face found
    if len(faces) == 0:
        continue
    first_face = faces[0]

    x, y, w, h = first_face
    face_frame = frame[y: y + h, x: x + w]
    preprocessed = preprocess_frame(face_frame)

    # Make a keypoints' locations prediction
    predictions = model.predict(preprocessed)[0]
    
    # Display a result in a new window
    result = cv.resize(face_frame, (96, 96))
    for i in range(0, len(predictions), 2):
        cv.circle(result, (predictions[i], predictions[i + 1]), 1, (255, 255, 0))

    result = cv.resize(result, (400, 400))
    cv.imshow('Predicted', result)

    # Perform untill 'q' key wasn't pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Call program closing methods
cap.release() 
cv.destroyAllWindows() 