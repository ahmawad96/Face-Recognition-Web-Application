import cv2
import numpy as np
from time import sleep

def DetectFace():
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    i=0
    while(True):
        _, frame=cap.read()
        frame_gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face= face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        if  np.any(face):
            if(i==0):
                sleep(3)
                i+=1
                continue
            for (x,y,w,h) in face:
                #cv2.rectangle(frame,(x,y-30),(x+w, y+h+30), (0,255,0),3)
                face_img=frame[y-30:y+h+30,x-40:x+w+40]

                #cv2.imwrite("face.jpg", face_img)
            break
    #     cv2.imshow("image",frame)
    #     #k = cv2.waitKey(30) & 0xff
    #     k = cv2.waitKey(30) & 0xff
    #     if k == 27:
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()
    return face_img

def prepare(image):
    h,w,_=image.shape
    mini=min(h,w)
    image=image[ :int(mini), :int(mini)]
    image=cv2.resize(image,(96,96))
    #cv2.imwrite("cropped.jpg",image)
    return image
