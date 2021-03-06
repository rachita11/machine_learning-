#predicting the rates of buildings based o data present about the locality and comparing with the original rates
#the project uses linear regression model to find the output beacuse all the features are linearly dependent
#the features in the dataset given represent the following information:-
#CRIM     per capita crime rate by town
#ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#INDUS    proportion of non-retail business acres per town
#CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#NOX      nitric oxides concentration (parts per 10 million)
#RM       average number of rooms per dwelling
#AGE      proportion of owner-occupied units built prior to 1940
#DIS      weighted distances to five Boston employment centres
#RAD      index of accessibility to radial highways
#TAX      full-value property-tax rate per $10,000
#PTRATIO  pupil-teacher ratio by town
#B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#LSTAT    % lower status of the population
#MEDV     Median value of owner-occupied homes in $1000's

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("buildings_ds.csv", sep="\s+")
data = data[["CRIM","CHAS","NOX","RM","DIS","RAD","TAX","PTRATIO","MEDV"]]
z= "MEDV"

x = np.array(data.drop([z],1))
y = np.array(data[z])

model = linear_model.LinearRegression()
model.fit(x,y)

print(model.intercept_)
print(model.coef_)

p = model.predict(x)

print("Value in $1000's")
for i in range(len(p)):
    print("prediction: ",p[i]," Real: ",y[i])

print("Accuracy=",(model.score(x,y)*100),"%")
