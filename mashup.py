import cv2
import numpy as np
import mediapipe as mp

import npwriter

name = input("Digite seu nome: ")

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True,
  max_num_hands=2, min_detection_confidence=0.5,
  min_tracking_confidence=0.5, model_complexity=0
)

mp_drawing = mp.solutions.drawing_utils

if len(name) == 0:
    print("Não foi possível atender sua solicitação: Nome inválido")
else:
    cap = cv2.VideoCapture(0)

    classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    f_list = []

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = classifier.detectMultiScale(gray, 1.5, 5)

        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

        faces = faces[:1]

        hands_captured = hands.process(frames_rgb)

        if len(faces) == 1:

            face = faces[0]

            x, y, w, h = face

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if not ret:
            continue
        
        if results.multi_hand_landmarks is not None:
            mp_drawing.draw_landmarks(frame, Landmarks, mp_hands.HAND_CONNECTIONS, circles_color, lines_color)

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
