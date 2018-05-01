# coding=UTF-8
import tensorflow as tf
from skimage.io import imsave
from skimage.transform import resize
import cv2
import os
import numpy as np

for filename in os.listdir('JPEGImages'):
    print(filename)
    image = cv2.imread('JPEGImages/'+filename)

    image = cv2.resize(image, (640, 360))
   
    imsave('results/' + filename, image)#储存图片
