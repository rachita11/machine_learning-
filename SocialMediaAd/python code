#project
#predicting if a person would purchase an item through social media marketing based on previos data behaviour
#using LOGISTIC REGRESSION

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("Social_Network_Ads.csv")
data = data[['Gender','Age','EstimatedSalary','Purchased']]

le = preprocessing.LabelEncoder()

Gender = le.fit_transform(list(data["Gender"]))
Age = le.fit_transform(list(data["Age"]))
Purchased = le.fit_transform(list(data["Purchased"]))
             
x = list(zip(Gender,Age,Purchased))
y= list(Purchased)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
model = LogisticRegression()
model.fit(X_train,y_train)

names=["Yes","No"]

predictions = model.predict(X_test)

for i in range(len(predictions)):
    print("Prediction:",names[predictions[i]]," Reality:",names[y_test[i]])
