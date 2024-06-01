from cv2 import (CascadeClassifier, VideoCapture, cvtColor,
                 resize as cvResize, COLOR_BGR2GRAY, rectangle,
                 INTER_CUBIC, imwrite, imshow,
                 putText, FONT_HERSHEY_COMPLEX,
                 waitKey, destroyAllWindows)
from cv2.data import haarcascades
from imutils import resize as imResize
from os.path import exists
from os import makedirs

personName = 'pyCampaDB'
personPath = f'assets/{personName}/'

if not exists(personPath):
    print('Creating the folder ' + personPath)
    makedirs(personPath)

cap = VideoCapture(0)
faceClassif = CascadeClassifier(haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = imResize(frame, width=640)
    gray = cvtColor(frame, COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)
    for (x, y, w, h) in faces:
        rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
        face = auxFrame[y:y+h, x:x+w]
        face = cvResize(face, (150, 150), interpolation=INTER_CUBIC)
        font = FONT_HERSHEY_COMPLEX
        putText(frame, personName, (x, y-10), font, 0.5, (0,0,255), 2)
        imwrite(personPath + f'/{personName}_{count}.jpg', face)
        count += 1

    imshow('Frame', frame)
    k = waitKey(1)

    if k == 27 or count >= 300: break

cap.release()
destroyAllWindows()