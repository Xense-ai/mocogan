import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video", type=str, 
	help="path to the input image")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)


if (cap.isOpened() == False): 
  print("Unable to read camera feed")


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))

while(True):
  ret, frame = cap.read()

  if ret == True: 
    

    out.write(frame)

    cv2.imshow('frame',frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

 
  else:
    break  


cap.release()
out.release()


cv2.destroyAllWindows()