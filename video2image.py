import cv2
vc=cv2.VideoCapture("video.mp4")
c=1  
rval,frame=vc.read()  
while rval:  
    rval,frame=vc.read()                  
    cv2.imwrite(('images/'+str(c)+'.jpg'),frame)
    c=c+1  
    cv2.waitKey(1)  
vc.release()
