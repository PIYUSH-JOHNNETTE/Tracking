import cv2
import numpy as np

# Initialize the Kalman filter
kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kf.processNoiseCov = 1e-5 * np.eye(4, dtype=np.float32)
kf.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)

# Initialize the KCF tracker
tracker = cv2.TrackerKCF_create()

# Read the first frame of the video
cap = cv2.VideoCapture(r"C:\Users\LENOVO\Downloads\Demo-video.mp4")
for i in range (0,23):
    ret, frame = cap.read()
ret, frame = cap.read()

# Initialize the object position
bbox = cv2.selectROI(frame, False)
kf.statePost = np.array([[bbox[0]], [bbox[1]], [0], [0]], dtype=np.float32)
ok = tracker.init(frame, bbox)
# Main loop for object tracking
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Predict the next state using the Kalman filter
    k=kf.predict()
    print(k)
    predicted_state = kf.statePre
    print(predicted_state)
    # Use the predicted state to initialize the tracker
    bbox = (int(predicted_state[0][0]), int(predicted_state[1][0]), bbox[2], bbox[3])
    
    
    # Update the Kalman filter and the tracker if the object is detected
    
        # Track the object using the KCF tracker
    ok, bbox = tracker.update(frame)
    if ok:
        
        # Update the Kalman filter based on the tracked position
        measurement = np.array([[bbox[0] + bbox[2]/2], [bbox[1] + bbox[3]/2]], dtype=np.float32)
        kf.correct(measurement)
        
        # Draw the bounding box around the tracked object
        bbox = tuple(map(int, bbox))
        cv2.rectangle(frame, bbox, (0, 255, 0), 2)
    else:
        # If the object is not detected, use the Kalman filter to predict its position
        predicted_bbox = (int(predicted_state[0][0]), int(predicted_state[1][0]), bbox[2], bbox[3])
        cv2.rectangle(frame, predicted_bbox, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
