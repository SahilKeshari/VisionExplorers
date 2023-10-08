import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model for eye and yawn detection
model_path = 'drowiness_new6.h5'  # Replace with the path to your .h5 model file
model = tf.keras.models.load_model(model_path)

eye_model_path = 'eyeModel.h5'  # Replace with the path to your .h5 model file
eye_model = tf.keras.models.load_model(eye_model_path)

input_shape = (145,145)
eye_input_shape = (64,64)

labels = ["yawn", "no_yawn", "Closed", "Open"]
# Define a function to process live frames and detect eyes and yawns
def process_frame(frame):
    yawn_frame = cv2.resize(frame, input_shape)
    prediction = model.predict(np.expand_dims(yawn_frame, axis=0))[0]

    if prediction[0]:
        cv2.putText(frame, "Yawn Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2)
    else:
        cv2.putText(frame, "No Yawn Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    eye_frame = cv2.resize(frame,eye_input_shape)
    eye_frame_gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    eye_frame_gray = eye_frame_gray / 255.0  # Normalize pixel values to [0, 1]
    eye_prediction = eye_model.predict(np.expand_dims(eye_frame_gray, axis=0))[0][0]

    print(eye_prediction)
    if eye_prediction > 0.5:
        cv2.putText(frame, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
    # Display the processed frame
    cv2.imshow("Result", frame)

# Open a video capture stream (0 is typically the default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    process_frame(frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
