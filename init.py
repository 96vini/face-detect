from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Acuracia do modelo:", accuracy)

data = [[5.1, 3.5, 1.4, 0.2],
       [6.2, 2.9, 4.3, 1.3],
       [7.3, 3.3, 6.0, 2.5]]

normalizedData = scaler.transform(data)

predict = knn.predict(normalizedData)

classnames = iris.target_names

predict_names = [classnames[classe] for classe in predict]

print("Previsoes");

for i, dt in enumerate(data):
    print(f"Dados: {dt} - Previsão: {predict_names[i]}")