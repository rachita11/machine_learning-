#This project uses "Decision Tree" model to classify wether a person has diabetes or not based on certain medical history
#using matplotlib library we get the final tree visualized

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from matplotlib import pyplot as plt

data = pd.read_csv("Diabetes.csv")

x = data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age"]]
y = data["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

model = DecisionTreeClassifier(criterion="entropy",max_depth=3)
model.fit(x_train,y_train)

predictions = model.predict(x_test)

#print(metrics.accuracy_score(y_test,predictions))

plt.figure(figsize=(15,10))
tree.plot_tree(model,feature_names=("Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age"),class_names=("0","1"),filled = True)
plt.show()
