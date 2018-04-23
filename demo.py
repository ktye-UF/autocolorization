# coding=UTF-8
import tensorflow as tf
from utils import *
from net import Net
from skimage.io import imsave
from skimage.transform import resize
import cv2
import os

for filename in os.listdir('test'):
	print(filename)
	img = cv2.imread('test/'+filename)
	if len(img.shape) == 3:
	  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	img = img[None, :, :, None]#batch1 
	data_l = (img.astype(dtype=np.float32)) / 255.0 * 100 - 50 # 灰度范围转化Lab范围并移至中�? 灰度[0,255], Lab[0,100]

	#data_l = tf.placeholder(tf.float32, shape=(None, None, None, 1))
	autocolor = Net(train=False)

	conv8_313 = autocolor.inference(data_l)

	saver = tf.train.Saver()
	with tf.Session() as sess:
	  tf.get_variable_scope().reuse_variables()
	  saver.restore(sess, 'models/model.ckpt')
	  conv8_313 = sess.run(conv8_313)

	img_rgb = decode(data_l, conv8_313, 2.63)
	imsave('output/' + filename, img_rgb)
