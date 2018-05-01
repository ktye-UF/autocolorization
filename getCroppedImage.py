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
    h = image.shape[0]#image 高度
    w = image.shape[1]#image 宽度

    if w > h:
      image = cv2.resize(image, (int(224 * w / h), 224))#转换成 resize：(image_size * w / h) * image_size ，按比例缩放成176 * 176
      crop_start = np.random.randint(0, int(224 * w / h) - 224 + 1)
      image = image[:, crop_start:crop_start + 224, :]#截取宽度为image_size .iamge_size * image_size 大小的图片
    else:
      image = cv2.resize(image, (224, int(224* h / w)))
      crop_start = np.random.randint(0, int(224 * h / w) - 224 + 1)
      image = image[crop_start:crop_start + 224, :, :]#截取高度为image_size .* image_size 大小的图片
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#转换图片BGR格式到RGB格式
    imsave('results/' + filename, image)#储存图片