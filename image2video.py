import cv2
import os

#图片路径
im_dir = 'result'
#输出视频路径
video_dir = '201708231503440-1018.avi'
#帧率
fps = 30 
#图片数 
num = 473
#图片尺寸
img_size = (640,360)

#fourcc = cv2.cv.CV_FOURCC('M','J','P','G')#opencv2.4
fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #opencv3.0
videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

for i in range(1,num):
    im_name = os.path.join(im_dir, str(i)+'.jpg')
    frame = cv2.imread(im_name)
    videoWriter.write(frame)
    print (im_name)

videoWriter.release()
print ('finish')
