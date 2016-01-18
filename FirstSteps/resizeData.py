import glob
import os
import sys

from skimage.transform import resize
from skimage.io import imread, imsave
import cv2
import numpy as np


#Set path of data files
INPUT_PATH = "/home/jeroen/Documents/kaggle/FirstSteps/"
size = 20
if len(sys.argv) > 1:
	size = int(sys.argv[1])
affix = ''
if len(sys.argv) > 2:
	affix = sys.argv[2]
	
def read_image(filename):	
	image = imread( filename )
#	image = cv2.imread(filename,0)
	if affix == 'GaussianAdapt':
		image = cv2.imread(filename,0)
		image = cv2.adaptiveThreshold(	image,255,
									cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            						cv2.THRESH_BINARY,11,2)
	elif affix == 'BilateralFiltering':
		image = cv2.bilateralFilter(image,9,75,75)
	elif affix == 'erode':
		kernel = np.ones((5,5),np.uint8)
		image = cv2.erode(image,kernel,iterations = 1)
	elif affix == 'dilate':
		kernel = np.ones((5,5),np.uint8)
		image =  cv2.dilate(image,kernel,iterations = 1)	
	elif affix == 'MorphologicalGradient':
		kernel = np.ones((5,5),np.uint8)
		image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)	
	elif affix:
		raise Exception('unknown affix')
	image = resize( image, (size,size) )	
	return image

train_output_path = INPUT_PATH + "/trainResized" + affix+str(size)
test_output_path = INPUT_PATH + "/testResized"+ affix+str(size)
if not os.path.exists( train_output_path ):
	os.makedirs( train_output_path )
if not os.path.exists( test_output_path ):
	os.makedirs( test_output_path )

##better os walk tree
trainFiles = glob.glob( INPUT_PATH + "/train/*" )
for i, nameFile in enumerate(trainFiles):
	print 'train', i, len(trainFiles)
	imageResized = read_image(nameFile)
	newName = os.path.sep.join( [train_output_path,os.path.split(nameFile)[-1]]	)
	imsave( newName, imageResized )

testFiles = glob.glob( INPUT_PATH + "/test/*" )
for i, nameFile in enumerate(testFiles):
	print 'test', i, len(testFiles)
	imageResized = read_image(nameFile)	
	newName = "/".join( [ test_output_path , os.path.split(nameFile)[-1] ])
	imsave ( newName, imageResized )