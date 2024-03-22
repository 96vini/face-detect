import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

# static_image_mode: check if is image or video
# max_num_hands: maximum hands to recognize
# min_detection_confidence and min_tracking_confidence: used to set percentage of confidence
# model_complexity: set complexity of model

hands = mp_hands.Hands(static_image_mode=True,
  max_num_hands=2, min_detection_confidence=0.5,
  min_tracking_confidence=0.5, model_complexity=0
)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
  ret, frames = cap.read()
  frames = cv2.flip(frames, 1)
  frames_rgb = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)

  results = hands.process(frames_rgb)

  print(results)

  circles_color = mp_drawing.DrawingSpec(color=(255,0,0), thickness=4, circle_radius=2)

  lines_color = mp_drawing.DrawingSpec(color=(0,0,255), thickness=3)
  print(results.multi_hand_landmarks)
  if results.multi_hand_landmarks is not None:
    for Landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(frames, Landmarks, mp_hands.HAND_CONNECTIONS, circles_color, lines_color)

      cv2.imshow("Landmarks", frames)

      t = cv2.waitKey(1)

      if t == ord('q') or t == ord('Q'):
          break
