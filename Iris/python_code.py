#The following code uses decision tree algorithm to predict the species of iris using the available dataset.


#importing all the necessary libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from matplotlib import pyplot as plt

#bringing the dataset to our code
data = pd.read_csv("iris.csv")
data = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']]

#seperating input and output from the data
x = np.array(data.drop(['Species'],axis=1))
y = np.array(data['Species'])

#split them into training and testing data (I am using about 80% for training the data)
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=1) 

#building the required model
model = DecisionTreeClassifier()
model.fit(X_train,y_train)

#visualize the desision tree formed by fitting our data
plt.figure(figsize=(10,9))
tree.plot_tree(model,feature_names=('SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'),class_names=("Iris-setosa","Iris-versicolor","Iris-virginica"),filled = True)
plt.show()

#predicting the label for our test data and print it out
predictions = model.predict(X_test)
for i in range(len(predictions)):
    print("Prediction:",predictions[i],"        Reality:",y_test[i])

#checking the accuracy of our model
acc = metrics.accuracy_score(predictions,y_test)
print("Accuracy=",acc)
