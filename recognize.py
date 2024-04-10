import cv2
import numpy as np
import pandas as pd
import operator

class KNN:

    def __init__(self, K=3):
        self.K = K

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            dist = np.array([self.euc_dist(np.resize(X_test[i], x_t.shape), x_t) for x_t in self.X_train])
            dist_sorted = dist.argsort()[:self.K]
            neigh_count = {}
            for idx in dist_sorted:
                if self.Y_train[idx] in neigh_count:
                    neigh_count[self.Y_train[idx]] += 1
                else:
                    neigh_count[self.Y_train[idx]] = 1
            sorted_neigh_count = sorted(neigh_count.items(), key=operator.itemgetter(1), reverse=True)
            if sorted_neigh_count[0][1] >= 5:
                predictions.append(sorted_neigh_count[0][0])
            else:
                predictions.append("não identificado")
        return predictions

    def euc_dist(self, x1, x2):
        x1 = np.array(x1, dtype=float)
        try:
            x2 = np.array(x2, dtype=float)
            return np.sqrt(np.sum((x1 - x2) ** 2))
        except ValueError:
            return float('inf')

f_name = "face_data.csv"

data = pd.read_csv(f_name).values

X, Y = data[:, 1:-1], data[:, -1]

# Knn função chamando com k = 5
model = KNN(K=5)

# Treinamento do modelo
model.fit(X, Y)

# Captura de vídeo
cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5)

    X_test = []  # Dados de teste
    for face in faces:
        x, y, w, h = face
        im_face = gray[y:y + h, x:x + w]
        im_face = cv2.resize(im_face, (100, 100))
        X_test.append(im_face.reshape(-1))

    if len(X_test) > 0:  # Verifica se pelo menos um rosto foi detectado
        response = model.predict(np.vstack(X_test))
        for i, face in enumerate(faces):
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(frame, str(response[i]), (x - 50, y - 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("full", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
