import cv2
import sys
from skimage.feature import match_descriptors, plot_matches, SIFT
import numpy as np
import ctypes
# Set up tracker.
# Instead of MIL, you can also use
 

tracker = cv2.TrackerKCF_create()
tracker1=cv2.legacy.TrackerKCF_create()    


# Read video
video = cv2.VideoCapture(r"D:\learning\los_angeles.mp4")
 
# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()
 
# Read first frame.
ok, frame = video.read()
# cv2.resize(frame,(80,40))
if not ok:
    print('Cannot read video file')
    sys.exit()
     
# Define an initial bounding box
# bbox = (287, 23, 86, 320)
# Uncomment the line below to select a different bounding box
bbox = cv2.selectROI(frame, False)

img=frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# sift= SIFT()
# kp = sift.detect_and_extract(img)
# keypointsOrig = sift.keypoints
# descriptorsOrig = sift.descriptors
sift=cv2.SIFT_create()
kp2, des2 = sift.detectAndCompute(img, None)
p=kp2[0]
img_with_keypoints = cv2.drawKeypoints(img, kp2, None)
cv2.imshow('Image with keypoints', img_with_keypoints)
# print(frame)
croped_frame =[]

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)


lis=[]
global l

l=[]
def update_frame():
   while True:
      success,frame=video.read()
      a=[]
      if not success:
         break
      else:
         a.append(frame) 
              
      time=cv2.getTickCount()
      a.append(time)

      # ok,bbox=tracker.update(frame)
      #print("Function Print", a)
      return a


def cropped_window(bbox,bool):
    print('cropped window')
    # if bbox[1]-100>0 and bbox[0]-100>0 :
    #      global cropped_frame 
    #      cropped_frame = frame[bbox[1]-100:bbox[1]+bbox[3]+100,bbox[0]-100:bbox[0]+bbox[2]+100]
    #      lis.append(bbox)
    #     #  cv2.imshow("Cropped Frame", cropped_frame)
    #   #   print(list)
    # elif bool :
    #      y=bbox[1]*100
    #      x=bbox[0]*100
    #      cropped_frame = frame[y-100:bbox[1]+bbox[3]+100,x-100:bbox[0]+bbox[2]+100]
        #  cv2.imshow("Cropped Frame", cropped_frame)
    return cropped_frame


def feature_extraction(cropped_frame):
   kp1, des1 = sift.detectAndCompute(cropped_frame, None)  
   matcher = cv2.FlannBasedMatcher()
   matches = matcher.match(des1, des2)
   matche=matches[0]
   src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
   dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
   M, mask = cv2.findHomography(src_pts,dst_pts, cv2.RANSAC, 5.0)
   matches_mask = mask.ravel().tolist()
   inlier_matches = []
   for i, match in enumerate(matches):
       if matches_mask[i] == 1:
          inlier_matches.append(match)
   percentage=(len(inlier_matches)/len(matches))*100
   img3 = cv2.drawMatches(cropped_frame, kp1, img, kp2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
   cv2.imshow("Inlier Matches", img3)
   # cv2.waitKey(0)
   print(percentage)
   return percentage


def track():
    
    while True:
     
      # Update tracker
      res=update_frame()
      frame=res[0] 
      ok,bbox=tracker.update(frame)
    #   creating cropped window
    #   img=cropped_window(bbox,ok)
    #   cv2.imshow("Cropped Frame", img)
      # if bbox[1]-100>0 and bbox[0]-100>0 and bbox[2]-100>0 and bbox[3]-100>0:
      #    print('if')
      global cropped_frame 
      print('before',bbox)
      # cropped_frame = frame[bbox[1]-60:bbox[1]+bbox[3]+100,bbox[0]-60:bbox[0]+bbox[2]+100]
      lis.append(bbox)
      # print('cropped',bbox)
      # cv2.imshow("Cropped Frame", cropped_frame)
      #    print('if',cropped_frame)
      # #   print(list)
      # else :
      #    print('else')
      #    print(bbox)
      #    # y=(bbox[1]-100)*-1
      #    # x=(bbox[0]-100)*-1
      #    cropped_frame = frame[bbox[1]:bbox[1]+bbox[3]+100,bbox[0]:bbox[0]+bbox[2]+100]
      #    cv2.imshow("Cropped Frame", cropped_frame)
      #    print('elif',cropped_frame)
      #  
      # print('#### ')
      # print(bbox)
      # print(cropped_frame)
      # print(frame)
      # print(len(frame))
      
      
      
      fps = cv2.getTickFrequency() / (cv2.getTickCount() - res[1])
   
      # Draw bounding box
      if ok:
         cropped_frame = frame[bbox[1]-60:bbox[1]+bbox[3]+100,bbox[0]-60:bbox[0]+bbox[2]+100]
         cv2.imshow("Cropped Frame", cropped_frame)
         l.append(frame)
         global h
         h = len(l)
         print(h)
         #   cv2.imshow('listframe',frame)
       
         # Tracking success
         p1 = (int(bbox[0]), int(bbox[1]))           
         p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
         # draw rectangle
         cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

      # elif feature_extraction(cropped_frame)>=2 and bbox[0]==0 and bbox[1]==0 and bbox[2]==0 and bbox[3]==0 :
      #    cv2.destroyAllWindows()
      #    bbox=lis[-2]
      #    fram=l[-2]
      #    print('elif',bbox)
      #    cropped_frame = frame[bbox[1]-60:bbox[1]+bbox[3]+100,bbox[0]-60:bbox[0]+bbox[2]+100]
      #    cv2.imshow("Cropped Frame", cropped_frame)
      #    cv2.imshow('Fram',fram)
      #    cv2.waitKey(0)
      #    # ok = tracker1.init(fram, bbox)
      #    #   ok, bbox = tracker.update(update_frame)
      #    print(h)
      #    video1 = cv2.VideoCapture(0)
      #    # video1.set(cv2.CAP_PROP_POS_FRAMES, h)
      #    ret, frame1 = video1.read()
      #    ok = tracker1.init(frame1, bbox)
      #    while ok:
      #       # print('problem')
      #       # print(frame)
      #       # print(len(frame))

      #       print("Before execution", res)
      #       cv2.imshow("frame1",frame1) 
      #       ok, bbox = tracker1.update(frame1)

         
      #       if ok:
      #                # Tracking success
      #          p1 = (int(bbox[0]), int(bbox[1]))
      #          p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        
      #          cv2.rectangle(frame1, p1, p2, (255,0,0), 2, 1)  
      else :
        # Tracking failure
        if feature_extraction(cropped_frame)>=2:
            cv2.destroyAllWindows()
            bbox=lis[-2]
            fram=l[-2]
            print('elif',bbox)
            cropped_frame = frame[bbox[1]-60:bbox[1]+bbox[3]+100,bbox[0]-60:bbox[0]+bbox[2]+100]
            cv2.imshow("Cropped Frame", cropped_frame)
            cv2.imshow('Fram',fram)
            cv2.waitKey(0)
            # ok = tracker1.init(fram, bbox)
            #   ok, bbox = tracker.update(update_frame)
            print(h)
            # video1 = cv2.VideoCapture(0)

            # video1.set(cv2.CAP_PROP_POS_FRAMES, h)
            # ret, frame1 = video1.read()
            ok = tracker1.init(fram, bbox)
            while True:
               # print('problem')
               # print(frame)
               # print(len(frame))
               # res=update_frame()
               # frame=res[0] 
               # video1 = cv2.VideoCapture(0)
               success,frame2=video.read()
               if not success:
                  break
               if success:
                  print("in else execution", frame2)
                  cv2.imshow("frame1",frame2) 
                  cv2.waitKey(0)                              
                  ok, bbox = tracker1.update(frame2)
                  # p1 = (int(bbox[0]), int(bbox[1]))
                  # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                  # print('p1',p1)
                  # print('p2',p2)
                  # cv2.rectangle(frame2, p1, p2, (255,0,0), 2, 1)
                  # print('exit')
                  if ok:
                           # Tracking success
                     p1 = (int(bbox[0]), int(bbox[1]))
                     p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                              
                     cv2.rectangle(frame2, p1, p2, (255,0,0), 2, 1)  
                     print('p1',p1)                 
                     print('p2',p2)
                     # cv2.waitKey(0)
                  cv2.imshow("Tracking", frame2)
               else: 
                   print('hiiii')      
                   cv2.putText(frame2, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                  #  print('hiiii')            
               # cv2.imshow("Tracking", frame2)
      # Display tracker type on frame
      cv2.putText(frame, "KCF Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
      
      # Display FPS on frame  
      cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
   
         # Display result
      cv2.imshow("Tracking", frame)
   
         # Exit if ESC pressed
      k = cv2.waitKey(40) & 0xff
      if k == 27 : 
            break
      
print('lets start')
track()
print(cropped_frame)
# while percentage!=0:
#         bbox=lis[len(lis)-1]
#         fram=l[len(l)-1]
#         ok = tracker.init(fram, bbox)
#         while True:
#             res=update_frame()
#             frame1=res[0]
#             # print('problem')
#             # print(frame)
#             # print(len(frame))
#             ok, bbox = tracker.update(frame1)
#             kp1, des1 = sift.detectAndCompute(cropped_frame, None)
#             matcher = cv2.FlannBasedMatcher()
#             matches = matcher.match(des1, des2)
#             src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#             dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#             M, mask = cv2.findHomography(src_pts,dst_pts, cv2.RANSAC, 5.0)
#             matches_mask = mask.ravel().tolist()
#             inlier_matches = []
#             for i, match in enumerate(matches):
#                if matches_mask[i] == 1:
#                 inlier_matches.append(match)
#             global percentage
#             percentage=(len(inlier_matches)/len(matches_mask))*100
#             if percentage>=20:
#                      print(percentage)