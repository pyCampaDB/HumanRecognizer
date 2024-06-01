from os import listdir
from os.path import dirname, abspath
from cv2 import imread
from cv2.face import LBPHFaceRecognizer_create
                    #FisherFaceRecognizer_create
                    #LBPHFaceRecognizer_create
#                   #EigenFaceRecognizer_create
from numpy import count_nonzero as npCountNonZero, array as npArray

dataPath = f'{dirname(abspath(__file__))}\\assets'

peopleList = listdir(dataPath)
print("People's list: ", peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = f'{dataPath}\\{nameDir}'
    print('Reading images from ', personPath)

    for fileName in listdir(personPath):
        print(f'Faces: {nameDir}\\{fileName}\n')
        labels.append(label)
        facesData.append(imread(f'{personPath}\\{fileName}', 0))
        img = imread(f'{personPath}\\{fileName}', 0)
        #imshow('image', img)
        #waitKey(1)
    label += 1
print(f'labels = {labels}')
print(f'Number of labels 0 = {npCountNonZero(npArray(labels) == 0)}')
print(f'Number of labels 1 = {npCountNonZero(npArray(labels) == 1)}')
print(f'Number of labels 2 = {npCountNonZero(npArray(labels) == 2)}')
print(f'Number of labels 3 = {npCountNonZero(npArray(labels) == 3)}')
print(f'Number of labels 4 = {npCountNonZero(npArray(labels) == 4)}')
#Training the faces recognized
#face_recognizer = EigenFaceRecognizer_create()
#face_recognizer = LBPHFaceRecognizer_create()
face_recognizer = LBPHFaceRecognizer_create()
print('Trainning...')
face_recognizer.train(facesData, npArray(labels))

#Save the model scored
#face_recognizer.write('modelEigenFace.xml')
face_recognizer.write('modelLBPHFace.xml')
#face_recognizer.write('modelFisherFace.xml')

#print('Model stored: modelEigenFace.xml')
print('Model stored: modelLBPHFace.xml')
#print('Model stored: modelFisherFace.xml')

