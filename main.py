import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


train = pd.read_csv('KDDTrain.csv', header=None)
test = pd.read_csv('KDDTest.csv', header=None)

print("Dataset Shape:", train.shape)

train = train.iloc[:, :-1]
test = test.iloc[:, :-1]


X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]


for col in [1, 2, 3]:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])


le_y = LabelEncoder()
le_y.fit(list(y_train) + list(y_test))

y_train = le_y.transform(y_train)
y_test = le_y.transform(y_test)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}


print("\nModel Results:\n")

for name, model in models.items():
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')

    print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")