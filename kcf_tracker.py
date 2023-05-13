import cv2
import numpy as np



# Create a VideoCapture object to read from the camera
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r"C:\Users\LENOVO\Downloads\Demo-video.mp4")


tracker = cv2.TrackerKCF_create()

sta = False
bbox=[]
i=1


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

    if not ret:
        print("Could not read frame from camera")
        break
    curr_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # curr_points = cv2.goodFeaturesToTrack(curr_frame, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', mouse_callback)
    if i==0:
       tracker.init(frame, bbox)
       sta=True
       i+=1
    #    print(i)
       continue 
    if sta:

        # tracker.init(frame, bbox)
         # Update the tracker with the new frame
        success, bbox = tracker.update(frame)
        # bbox=int(bbox)


        # print('start')
        # Draw the bounding box around the tracked object
        if success:
            (x, y, w, h) = [int(v) for v in bbox]


            
            ###########kcf rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) 
            # cv2.circle(frame, (int(z_pred[0]), int(z_pred[1])), 5, (0, 0, 255), -1)
        else:
           
                    # If the object is not detected, use the predicted state vector and measurement vector as a placeholder
            
      
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