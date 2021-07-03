#According to given data, predicting if a person has possibly survived or not and comparing it with real results
#the project uses "K Nearest Neighbors" model for doing the classification

import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model,preprocessing

data = pd.read_csv("titanic_ds.csv")
data = data[["Survived","Pclass","Sex","Age"]]

le = preprocessing.LabelEncoder()

Survived = le.fit_transform(list(data["Survived"]))
Pclass = le.fit_transform(list(data["Pclass"]))
Age = le.fit_transform(list(data["Age"]))
Sex = le.fit_transform(list(data["Sex"]))

x = list(zip(Pclass,Sex,Age))
y = list(Survived)

model = KNeighborsClassifier(n_neighbors=3)

x_test,x_train,y_test,y_train = sklearn.model_selection.train_test_split(x,y,test_size=0.3)

model.fit(x_train,y_train)

names = ["YES","NO","UC"]
#UC = unclear

predictions = model.predict(x_test)
for i in range(len(predictions)):
    print("Prediction: ",names[predictions[i]], " Reality:",names[y_test[i]])
