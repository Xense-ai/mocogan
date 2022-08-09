import argparse
import imutils
import cv2
import numpy as np
import os
import mediapipe as mp

def difference(length,minl,maxl):
     return int(length - (maxl - minl))

def set_ranges(diff):
    if diff % 2 == 0:
        return int(diff/2),int(diff/2)
    else:
        return int((diff/2)+0.5),int((diff/2)-0.5)
    
def check_range(length,minl,maxl):
    if length == 0:
        if minl < length:
            temp = minl - length
            return length, maxl - temp
        else:
            return minl,maxl
    else:
        if maxl > length:
            temp = maxl - length
            return minl - temp,length
        else:
            return minl,maxl
    
mp_pose = mp.solutions.pose

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, 
	help="path to the input image")
ap.add_argument("-d","--div",type =int,
    help="integer for dividing")
ap.add_argument("-r",'--rotate',type = int,
    help = "rotate by x degrees clockwise")
ap.add_argument("-p",'--path',type = str,
    help = 'path to write jpg to')
args = vars(ap.parse_args())

images = []
counter = 0
cap = cv2.VideoCapture(args["video"])  
div = args["div"]
rotation = args["rotate"]
path = args['path']


    
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose: 
    
  x1=[]
  x2=[]
  y1=[]
  y2=[]
  
  while cap.isOpened():
    success, image = cap.read()
    
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break
  
    if rotation == 90:
        image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        image = cv2.rotate(image,cv2.ROTATE_180)
    elif rotation ==270:
        image = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    results = pose.process(image)   
    height, width, channels = image.shape   
    if results.pose_landmarks:
       
        xs=[]
        ys=[]

        for i in range(33):
            xs.append(int(results.pose_landmarks.landmark[i].x*width))
            ys.append(int(results.pose_landmarks.landmark[i].y*height))
        x1.append(min(xs))
        x2.append(max(xs))
        y1.append(min(ys))
        y2.append(max(ys))

        minx = min(x1)
        maxx = max(x2)
        miny = min(y1)
        maxy = max(y2)
        
if width < height:
    main = width
else:
    main = height
    
diffw = difference(main,minx,maxx) 
diffh = difference(main,miny,maxy)

height_1,height_2 = set_ranges(diffh)
width_1,width_2 = set_ranges(diffw)
    
miny = int(miny - height_1)
maxy = int(maxy + height_2)
minx = int(minx - width_1)
maxx = int(maxx + width_2)
    
if width < height:
    minx,maxx = check_range(0, miny, maxy)
    minx,maxx = check_range(main,miny,maxy)
else:
    miny,maxy = check_range(0, miny, maxy)
    miny,maxy = check_range(main,miny,maxy)


   
cap = cv2.VideoCapture(args["video"])   
while cap.isOpened():
    success, image = cap.read()
    
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break
    
    if rotation == 90:
        image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        image = cv2.rotate(image,cv2.ROTATE_180)
    elif rotation ==270:
        image = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    cropped = image[miny:maxy,minx:maxx] 
    resized = imutils.resize(cropped, width=128,inter = 2)

    
    if counter % div == 0:
        images.append(resized)
    counter += 1
   
cv2.imshow('s',cropped) 
cv2.waitKey(1000)
capname = args["video"][36:-4]
filename = "%s.jpg"% capname

concat = np.concatenate(images,axis=1)
cv2.imshow('concatenated',concat)
height, width, channels = concat.shape
print(width,height)
print(width/height)
cv2.imwrite(os.path.join(path,filename),concat)
cv2.waitKey(1000)
cap.release()
cv2.destroyAllWindows()    