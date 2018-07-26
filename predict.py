import numpy as np
import os
from scipy import  misc
from keras.models import model_from_json
import pickle
import cv2
import imutils

#def myMain():

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

classifier_f = open("int_to_word_out.pickle", "rb")
int_to_word_out = pickle.load(classifier_f)
classifier_f.close()

# load json and create model
json_file = open('model_face.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_face.h5")
print("Model is now loaded in the disk")

lis = os.listdir("/home/joker/Desktop/image_classification/predict")
for i in range(0,len(lis)):

	img = os.listdir("predict")[i]
	img1 = "predict/"+img
	input1 = cv2.imread(img1)
	
	image = np.array(misc.imread("predict/"+img))
	image = misc.imresize(image, (64, 64))
	image = np.array([image])
	image = image.astype('float32')
	image = image / 255.0

	prediction=loaded_model.predict(image)
	#print(prediction)
	j=np.max(prediction)
	print(j)
	print(int_to_word_out[np.argmax(prediction)])
	input1 = imutils.resize(input1, width=250)
	label = "{}: {:.2f}%".format(int_to_word_out[np.argmax(prediction)], j*100)
	cv2.putText(input1,label, (3,9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
	cv2.imshow("Input",input1)
	cv2.waitKey(1000)
	#f=open('output.txt','w')
	#f.close()