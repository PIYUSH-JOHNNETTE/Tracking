import cv2
import numpy as np


# create an ORB detector object
# orb = cv2.ORB_create(scaleFactor=0.02)

# Define the state vector x and the measurement vector z
# In this example, we assume the object's motion is described by a 4D state vector (x, y, vx, vy)
# and the measurements are 2D position measurements (x, y)
x = np.array([0, 0, 0, 0], dtype=np.float32)
z = np.array([0, 0], dtype=np.float32)

# Define the state transition matrix A and the measurement matrix H
A = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], dtype=np.float32)
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]], dtype=np.float32)

# Define the process noise covariance matrix Q and the measurement noise covariance matrix R
Q = np.array([[0.01, 0, 0.02, 0],
              [0, 0.01, 0, 0.02],
              [0.02, 0, 0.04, 0],
              [0, 0.02, 0, 0.04]], dtype=np.float32)
R = np.array([[1, 0],
              [0, 1]], dtype=np.float32)

# Initialize the Kalman filter
kf = cv2.KalmanFilter(4, 2, 0)
kf.transitionMatrix = A
kf.measurementMatrix = H
kf.processNoiseCov = Q
kf.measurementNoiseCov = R
kf.statePost = x

# Initialize the predicted state vector x_pred and the predicted measurement vector z_pred
x_pred = np.zeros_like(x)
z_pred = np.zeros_like(z)










# Create a VideoCapture object to read from the camera
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r"C:\Users\LENOVO\Downloads\Demo-video.mp4")


tracker = cv2.TrackerCSRT_create()

sta = False
bbox=[]
i=1
# global prev_frame



def get_new_coordinates(coordinate,  new_frame_height, new_frame_width, old_frame_height = 480, old_frame_width = 640):
        # Calculate the ratio of the old and new frames
        width_ratio = new_frame_width / old_frame_width
        height_ratio = new_frame_height / old_frame_height
        
        # Calculate the new coordinates using the ratio
        new_x = int(coordinate[0] * width_ratio)
        new_y = int(coordinate[1] * height_ratio)
        
        return (new_x, new_y)





# Define the callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    global sta,bbox,i,prev_points
    global prev_frame
    if event == cv2.EVENT_LBUTTONUP:
        i=0        
        print("Selected coordinates:", x, y)
        # Draw a circle at the selected point
        # cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        x1=(x-10)
        y1=(y-10)
        bbox=[x1,y1,20,20]
        img=frame[y1:y1+20,x1:x1+20]
        cv2.imshow('img',img)
        print(bbox)
        # print(img)
        # print(frame)

# Check if the camera was successfully opened
if not cap.isOpened():
    print("Could not open camera")
    exit()
# l=False
# Loop to read frames from the camera
while True:
    
    # Read a frame from the camera
    ret, frame = cap.read()
    frame=cv2.resize(frame,(640,480))
    # if l==True:
    #     l=False
    #     print('---------------')
    # print('between')
    # Check if the frame was successfully read
    if not ret:
        print("Could not read frame from camera")
        break
    curr_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # curr_points = cv2.goodFeaturesToTrack(curr_frame, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', mouse_callback)
    if i==0:
       tracker.init(curr_frame, bbox)
       sta=True
       i+=1
    #    print(i)
       continue 
    if sta:

        # tracker.init(frame, bbox)
         # Update the tracker with the new frame
        success, bbox = tracker.update(curr_frame)
        # bbox=int(bbox)
        print(bbox)

        # print('start')
        # Draw the bounding box around the tracked object
        if success:
            (x, y, w, h) = [int(v) for v in bbox]
            z = np.array([x,y], dtype=np.float32)
            # Update the Kalman filter with the measurement
            kf.correct(z)
            # Update the predicted state vector and measurement vector based on the filtered state
            x_pred = kf.predict()
            # print(x_pred)
            z_pred = np.dot(H, x_pred)
            # print(z_pred)

            cv2.circle(frame, (int(x_pred[0]), int(x_pred[1])), 5, (0, 0, 255), -1)
            ###########kcf rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) 
            # cv2.circle(frame, (int(z_pred[0]), int(z_pred[1])), 5, (0, 0, 255), -1)
        else:
           
                    # If the object is not detected, use the predicted state vector and measurement vector as a placeholder
            x_pred = kf.predict()
            z_pred = np.dot(H, x_pred)
            cv2.circle(frame, (int(z_pred[0]), int(z_pred[1])), 5, (0, 0, 255), -1)     
      
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

            # print('hiii')

        # Display the frame in a window
    
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# cv2.setMouseCallback('frame', mouse_callback)
# Clean up
cap.release()
cv2.destroyAllWindows()
