import os
import cv2

img_root = 'results/'#这里写你的文件夹路径，比如：/home/youname/data/img/,注意最后一个文件夹要有斜杠
fps = 24    #保存视频的FPS，可以适当调整

#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('saveVideo.avi',fourcc,fps,(224,224))#最后一个是保存图片的尺寸

for i in range(49):
    frame = cv2.imread(img_root+str(i+1)+'.jpg')
    videoWriter.write(frame)
videoWriter.release()
