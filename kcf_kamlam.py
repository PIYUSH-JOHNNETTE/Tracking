import cv2
import numpy as np

# create Kalman filter object
kalman = cv2.KalmanFilter(4, 2)
# state variables: x, y, dx/dt, dy/dt
# measurement variables: x, y
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1e-3, 0, 0, 0], [0, 1e-3, 0, 0], [0, 0, 1e-3, 0], [0, 0, 0, 1e-3]], np.float32) * 0.03

# create video capture object
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r"C:\Users\LENOVO\Downloads\Demo-video.mp4")

# detect feature point in first frame
for i in range(0,23):
    ret,frame=cap.read()
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_keypoints = cv2.goodFeaturesToTrack(gray, 1, 0.01, 10)
print(prev_keypoints)
prev_point = prev_keypoints[0]
print(prev_point)

# initialize Kalman filter state
kalman.statePre = np.array([prev_point[0][0], prev_point[0][1], 0, 0], np.float32)
kalman.statePost = np.array([prev_point[0][0], prev_point[0][1], 0, 0], np.float32)

while True:
    l=[]
    # read next frame from video
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # predict next state with Kalman filter
    kalman.predict()

    # detect feature point in current frame
    curr_keypoints = cv2.goodFeaturesToTrack(gray, 1, 0.01, 10)
    if curr_keypoints is not None:
        curr_point = curr_keypoints[0]

        # update Kalman filter with measured state
        kalman.correct(np.array([curr_point[0][0], curr_point[0][1]], np.float32))

        # get predicted state
        predicted_point = kalman.predict()[0:2]
        l.append(predicted_point)
        print(predicted_point)
        a=prev_point[0]
        b=curr_point[0]
        print('------------------------------',a,b)
        # draw lines showing predicted and measured points
        cv2.line(frame, (int(a[0]),int(a[1])), (int(predicted_point[0]),int(predicted_point[1])), (0, 0, 255), 2)
        cv2.line(frame, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])), (0, 255, 0), 2)

        # update previous point
        prev_point = l  

    # display frame
    cv2.imshow('frame', frame)
    # cv2.waitKey(0)
    if cv2.waitKey(1) == 27: # exit if 'esc' key is pressed
        break

# release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
