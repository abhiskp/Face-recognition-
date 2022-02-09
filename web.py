from datetime import datetime
import cv2
import numpy as np
import face_recognition
import os

path = 'D:\Software\pyProjects\ImageBasic'
imgs = []
classNames = []
myList = os.listdir(path)
#print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    imgs.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def fndEncod(imgs):
    encodeList=[]
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def EnterReg(name):
    with open('reg.csv','r+') as f:
        myDlist = f.readlines()
        nList = []
        for line in myDlist:
            entr = line.split(',')
            nList.append(entr[0])
        if name not in nList:
            now = datetime.now()
            dtStr = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtStr}')



encodelistKnown = fndEncod(imgs)
print('Encoding complete')


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrFr = face_recognition.face_locations(imgS)
    encodesCurrFr = face_recognition.face_encodings(imgS, facesCurrFr)

    for encodeFace,faceLoc in zip(encodesCurrFr, facesCurrFr):
        m = face_recognition.compare_faces(encodelistKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodelistKnown, encodeFace)
        print(faceDis)
        matchInd = np.argmin(faceDis)

        if m[matchInd]:
            name = classNames[matchInd].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = 4*y1,4*x2,4*y2,4*x1
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,x1,y2-35),(x2,y2,(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            EnterReg(name)

    cv2.imshow('webcam', img)


    cv2.waitKey(1)