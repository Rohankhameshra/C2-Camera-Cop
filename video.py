# import the necessary packages
from imutils.video import VideoStream
from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD
from imagenet_utils import preprocess_input
from keras import applications
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

top_model_weights_path = 'bottleneck_fc_model.h5'
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["Robber", "Non-Robber"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
model = applications.VGG16(include_top=False, weights='imagenet')


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
frame = vs.read()
frame = imutils.resize(frame, height=224,width=224)
cv2.imshow("Frame", frame)
image = image_utils.load_img("./data/Train/nonrobber/245.jpg", target_size=(224, 224))
image = image_utils.img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)
preds = model.predict(image)
print (preds.shape[1:])
model1 = Sequential()
model1.add(Flatten(input_shape=preds.shape[1:]))
model1.add(Dense(512, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(512, activation='relu'))
model1.add(Dropout(0.5 ))
model1.add(Dense(1, activation='sigmoid'))
model1.load_weights(top_model_weights_path)
preds = model1.predict(preds)
print (len(preds.shape))
print (preds.shape[1])
print preds
if preds>0.5:
	label= "robber"
else:
	label= "non-robber"
print ("Label: %s"%label)
'''
(inID, label) = decode_predictions(preds)[0]
print("Label: {}".format(label))
print (preds)
print ("confidance %f"%preds[0][np.argmax(preds, axis=-1)])

while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, height=224,width=224)
	image = image_utils.img_to_array(frame)
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)
	preds = model.predict(image)
	preds = model1.predict(preds)
	(inID, label) = decode_predictions(preds)[0]
	print("Label: {}".format(label))
	print (preds)
	print ("confidance %f"%preds[0][np.argmax(preds, axis=-1)])
	cv2.putText(frame, "Label: {}".format(label), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()'''
vs.stop()