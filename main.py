import numpy as np
import face_recognition
import os
from datetime import datetime
import cv2

path = 'Images'
images = []
classNames = []
imgList = os.listdir(path)
print(imgList)
for cl in imgList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as file:
        dataList = file.readlines()
        nameList = []
        for line in dataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            file.writelines(f'\n{name}, {dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

capture = cv2.VideoCapture(0)

while True:
    sucess, img = capture.read()
    imgSize = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSize = cv2.cvtColor(imgSize, cv2.COLOR_BGR2RGB)

    facesOnFrame = face_recognition.face_locations(imgSize)
    encodesOnFrame = face_recognition.face_encodings(imgSize, facesOnFrame)

    for encodeFace, faceLocation in zip(encodesOnFrame, facesOnFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
