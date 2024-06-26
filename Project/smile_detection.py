# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model("model/trained_model.h5")
camera = cv2.VideoCapture("videos/smiling_vid.mp4")

smile_count = 0
total_frame_count = 0
	
while True:
	# grab the next frame
	(grabbed, frame) = camera.read()

	if not grabbed:
		break
	
	total_frame_count += 1	

	# resize the frame to 300 pixels 
	frame = imutils.resize(frame, width=300) 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert it to grayscale
	frameClone = frame.copy() # and then clone the original frame so we can draw on it later in the program

    # detect faces in the input frame, then clone the frame so that
	# we can draw on it
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over the face bounding boxes
	for (fX, fY, fW, fH) in rects:
		# extract the ROI of the face from the grayscale image,
		# resize it to a fixed 28x28 pixels, and then prepare the
		# ROI for classification via the CNN
		roi = gray[fY:fY + fH, fX:fX + fW]
		roi = cv2.resize(roi, (28, 28))
		roi = roi.astype("float") / 255.0 # Normalization to values btwn 0 n 1
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)

        # determine the probabilities of both "smiling" and "not
		# smiling", then add to smile_count
		(notSmiling, smiling) = model.predict(roi)[0]
		if smiling > notSmiling:
			smile_count += 1 

smile_ratio = (smile_count/total_frame_count)*100		
if smile_ratio >= 20:
	print("Smiling well.", smile_ratio, smile_count,total_frame_count)
else:
	print("Need to smile more.", smile_ratio, smile_count,total_frame_count)

# To Run: python smile_detection.py --cascade haarcascade_frontalface_default.xml --model model/trained_model.h5 --video videos/smiling_vid.mp4