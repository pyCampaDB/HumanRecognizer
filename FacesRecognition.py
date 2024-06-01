from cv2.face import LBPHFaceRecognizer_create #LBPHFaceRecognizer_create #EigenFaceRecognizer_create #FisherFaceRecognizer_create  #
from cv2.data import haarcascades
from cv2 import (VideoCapture,  CascadeClassifier,
                 VideoWriter_fourcc,VideoWriter,
                  cvtColor, COLOR_BGR2GRAY, INTER_CUBIC,
                 resize as cvResize, putText, LINE_AA,
                 rectangle, imshow, FONT_HERSHEY_COMPLEX,
                 waitKey,
                 destroyAllWindows)
from datetime import datetime
from time import time
from os import listdir, makedirs
from os.path import abspath, dirname, exists




if __name__ == "__main__":
    path = abspath(__file__)

    pathAssets = dirname(path)+ "\\assets"
    imagePath = listdir(pathAssets)
    #print(imagePath)

    #face_recognizer = EigenFaceRecognizer_create()
    face_recognizer = LBPHFaceRecognizer_create()
    #face_recognizer = FisherFaceRecognizer_create()

    #face_recognizer.read('modelEigenFace.xml')
    face_recognizer.read('modelLBPHFace.xml')
    #face_recognizer.read('modelFisherFace.xml')

    cap = VideoCapture(0)
    faceClassified = CascadeClassifier(
                        haarcascades +
                        'haarcascade_frontalface_default.xml')

    frame_size = (int(cap.get(3)), int(cap.get(4)))
    fourcc = VideoWriter_fourcc(*"XVID")

    detection = False
    detection_stopped_time = None
    timer_started = False

    while True:
        ret, frame = cap.read()
        if ret == False: break
        gray = cvtColor(frame, COLOR_BGR2GRAY)
        auxFrame = gray.copy()
        SECONDS_TO_RECORD_AFTER_DETECTION = 3

        faces = faceClassified.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = auxFrame[y: y + h, x:x + w]
            face = cvResize(face, (150, 150),
                            interpolation=INTER_CUBIC)

            result = face_recognizer.predict(face)
            """fontFace = FONT_HERSHEY_COMPLEX
            putText(frame, f'{result}', (x, y - 5),
                    fontFace, 0.1, (255, 0, 0), LINE_AA)"""
            count = 0
            # count2 = 0

            # print(result)
            if result[0] < 5800:
                putText(frame, f'{imagePath[result[0]]}',
                        (x, y - 25), 2, 1.1,
                        (0, 255, 0), 1, LINE_AA)
                rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                count += 1
            else:
                putText(frame, 'Desconocido', (x, y - 20),
                        2, 0.8, (0, 0, 255),
                        1, LINE_AA)
                rectangle(frame, (x, y), (x + w, y + h),
                          (0, 0, 255), 2)
                # imwrite(f'{pathAssets}\\No Recognition\\face_{count2}.jpg', frame_size)
                # count2 += 1

        if len(faces) > 0:
            if detection:
                timer_started = False
            else:
                detection = True
                current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                if not exists(f'{dirname(path)}\\Recognition'):
                    makedirs(f'{dirname(path)}\\Recognition')

                """out = VideoWriter(f'{dirname(path)}\\Recognition\\{current_time}_ModelEigenFace.mp4',
                                  fourcc, 20, frame_size)"""
                out = VideoWriter(f'{dirname(path)}\\Recognition\\{current_time}_ModelLBPHFace.mkv',
                                  fourcc, 20, frame_size)
                """out = VideoWriter(f'{dirname(path)}\\Recognition\\{current_time}_ModelFisherFace.mp4',
                                  fourcc, 20, frame_size)"""
                print('Started Recording!')

        elif detection:
            if timer_started:
                """ 
                            if your camera doesn't detect your face for three seconds, 
                            stop recording and write the file
                """
                if time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                    detection = False
                    timer_started = False
                    out.release()
                    print('Stop Recording!')
            else:
                timer_started = True
                detection_stopped_time = time()

        if detection:
            out.write(frame)


        imshow('frame', frame)

        k = waitKey(1)
        if k == 27:
            break
    out.release()
    cap.release()
    destroyAllWindows()