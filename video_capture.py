import cv2
import os
time = 500
print(cv2.__version__)
vidcap = cv2.VideoCapture('2018_0410_214530_001_EVE.MOV')
success,image = vidcap.read()
count = 0
path = './Images'
success = True
while success:
	vidcap.set(cv2.CAP_PROP_POS_MSEC,time)
	cv2.imwrite(os.path.join(path,"frame1%d.jpg" % count), image)     # save frame as JPEG file
	success,image = vidcap.read()
	print ('Read a new frame: ', success)
	count += 1
	time += 500

