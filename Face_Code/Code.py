import numpy as np
import cv2
import pickle
from PIL import Image

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
k=1
labels = {"person_name" : 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

#cap=cv2.VideoCapture(0)
frame = cv2.imread("ex.jpg")
while(k==1):
    #ret, frame = cap.read()
    #frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    for(x, y, w, h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y + h, x:x + w]
        id_,conf = recognizer.predict(roi_gray)
        print(conf)
        if conf>=10 and conf<=100:
            print(id_)
            print(labels[id_])
            print (conf)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            opp = str(round(conf))
            color = (0,0,244)
            #cv2.putText(frame, opp, (x,y+h+25), font, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, name, (x,y-8), font, 1.5, color, 2, cv2.LINE_AA)

        k=0
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_color)

        color = (255,0,0)
        cv2.rectangle(frame,(x,y),(x+w,y+h), color,2)
        eyes = eye_cascade.detectMultiScale(roi_gray)



image = image_resize(frame, height = 450)
while(True):
    cv2.imshow('frame', image)
    if(cv2.waitKey(20)&0xFF==ord('q')):
        break