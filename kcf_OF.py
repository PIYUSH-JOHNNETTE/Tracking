import cv2
import numpy as np
from skimage.metrics import structural_similarity

# create an ORB detector object
# orb = cv2.ORB_create(scaleFactor=0.02)

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
        print(bbox)
        # print(img)
        # print(frame)
        cv2.imshow('prev_frame',img)
        
        # cv2.waitKey(0)
        # prev_frame=frame
        prev_frame=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prev_points = cv2.goodFeaturesToTrack(prev_frame, maxCorners=100, qualityLevel=0.2, minDistance=20, blockSize=7, useHarrisDetector=False, k=0.04)
        # prev_points=get_new_coordinates()
        # # detect ORB keypoints in the first frame
        # prev_keypoints = orb.detect(prev_frame, None)
        # print(prev_keypoints)
        # # compute the descriptors for the detected keypoints
        # prev_keypoints, prev_descriptors = orb.compute(prev_frame, prev_keypoints)

        # # create an array of initial points for optical flow
        # prev_points = cv2.KeyPoint_convert(prev_keypoints)

        print(prev_points)
        print(prev_frame.shape)
        print('--------------------',frame.shape)

# Check if the camera was successfully opened
if not cap.isOpened():
    print("Could not open camera")
    exit()
# l=False
# Loop to read frames from the camera
while True:
    print('wlcome')
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
            img1=frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
            i_h,i_w,_=img1.shape
            # curr_frame=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            # print(curr_frame.shape)
            if prev_frame.shape != curr_frame.shape:
                shap=prev_frame.shape
                curr_frame=cv2.resize(curr_frame,(shap[1],shap[0]))
            # print(curr_frame.shape)
            # curr_points = cv2.goodFeaturesToTrack(curr_frame, maxCorners=100, qualityLevel=0.5, minDistance=20, blockSize=7)
            # print(curr_points) 

            p1, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_points, None)
             
            print('p1------------------------',p1)

            print(status)
            
            # print(status[1][0])
            # for i in range(0,3):
            #     print(p1)
            #     print(status)
            #     coor=p1[i]
            #     print(coor)
                
            #     i+=1
            #     if i==3:
            #         exit 
            (x, y, w, h) = [int(v) for v in bbox]
            for j in range (0,len(status)):
                # print(p1[0][0][0])
                if  status[j]==1:
                   #optical flow rectangle 
                    coor=p1[j][0]
                    print(coor)
                    a=int(coor[0])
                    b=int(coor[1])
                    print(a)
                    print(b)
                    # new_coor1=get_new_coordinates((a,b),640,480,i_h,i_w)
                    # print('###############################',new_coor1)
                    cv2.rectangle(img1, (a-10, b-10), ( a+10, b+10), (0, 255, 0), 2)
                    new_coor=get_new_coordinates((x,y),i_h,i_w)
                    print(new_coor)
                    # x1=int((new_coor[0]+(a-10))/2)
                    x1=int(((0.3*new_coor[0])+(0.7*a))/2)
                    y1=int((new_coor[1]+b)/2) 
                    print(x1,y1)
                    print(x,y)
                    # new_coor1=get_new_coordinates((x1,y1),640,480,i_h,i_w)
                    # cv2.rectangle(frame, (new_coor1[0]-10, new_coor1[1]-10), ( new_coor1[0]+10, new_coor1[1]+10), (0, 0, 0), 2)
                    cv2.rectangle(img1, (x1-6, y1-6), ( x1+8, y1+8), (0, 0, 0), 2)
                    cv2.imshow ('img1',img1)
                    # cv2.circle(frame,(a,b),1,(0,0, 255), 2)  
                    cv2.circle(img1,(a,b),1,(0,130, 255), 2)
            ###########kcf rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) 
            prev_frame=curr_frame
            # prev_points=curr_points   
        else:
           
        #    for value in status:
        #        if value==1:
        #          print('hiiiii')
        #          prev_frame=curr_frame
        #          prev_points=curr_points
        #          l=True
        #          break
                 
        #        else:
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
