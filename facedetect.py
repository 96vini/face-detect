import cv2
import numpy as np

import npwriter

name = input("Digite seu nome: ")

if len(name) == 0:
    print("Não foi possível atender sua solicitação: Nome inválido")
else:
    cap = cv2.VideoCapture(1)

    classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    f_list = []

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(gray, 1.5, 5)

        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

        faces = faces[:1]

        if len(faces) == 1:

            face = faces[0]

            x, y, w, h = face

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if not ret:
            continue

        cv2.imshow("full", frame)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('c'):
            if len(faces) == 1:
                gray_face = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (100, 100))
                print(len(f_list), type(gray_face), gray_face.shape)
                # this will append the face's coordinates in f_list
                f_list.append(gray_face.reshape(-1))
            else:
                print("face not found")

            # this will store the data for detected
            # face 10 times in order to increase accuracy
            if len(f_list) == 10:
                break

    npwriter.write(name, f_list)
    cap.release()
    cv2.destroyAllWindows()
