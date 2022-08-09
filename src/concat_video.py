import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-s", "--speed", required=True ,type = int,
	help="speed of video, 0 gives a frame by frame")
args = vars(ap.parse_args())

image = cv2.imread(args["image"],0)
speed = args["speed"]

h = image.shape[0]
w = image.shape[1]

frames = w/h

for i in range(frames-1):
    display= image[0:h,0:h]
    image= image[0:h,h:w]
    cv2.imshow('video',display)
    if speed == 0:
        cv2.waitKey(0)
    else:
        cv2.waitKey(speed)