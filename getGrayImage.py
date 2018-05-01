# coding=UTF-8
import tensorflow as tf
import cv2
import os
from skimage.io import imsave

for filename in os.listdir('JPEGImages'):
	print(filename)
	img = cv2.imread('JPEGImages/'+filename)
	if len(img.shape) == 3:
	  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	  
	imsave('results/' + filename, img)
