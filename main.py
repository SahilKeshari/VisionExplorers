import cv2
import numpy as np
import tensorflow as tf


model_path = 'drowsiness.h5'  
model = tf.keras.models.load_model(model_path)

eye_model_path = 'eyeModel.h5'  
eye_model = tf.keras.models.load_model(eye_model_path)

input_shape = (145,145)
eye_input_shape = (64,64)

def process_frame(frame):
    yawn_frame = cv2.resize(frame, input_shape)
    prediction = model.predict(np.expand_dims(yawn_frame, axis=0))[0]

    if prediction[0]:
        cv2.putText(frame, "Yawn Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2)
    else:
        cv2.putText(frame, "No Yawn Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    eye_frame = cv2.resize(frame,eye_input_shape)
    eye_frame_gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    eye_frame_gray = eye_frame_gray / 255.0 
    eye_prediction = eye_model.predict(np.expand_dims(eye_frame_gray, axis=0))[0][0]

    print(eye_prediction)
    if eye_prediction > 0.5:
        cv2.putText(frame, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Result", frame)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    process_frame(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
