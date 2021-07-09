import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv("SampleSuperstore.csv") 

#getting information about the data we have
print(data.dtypes)

#getting geographical insights
print(data['Region'].unique(),'\n')
print(data['Country'].unique(),'\n')
print(data['State'].unique(),'\n')
print(data['City'].unique(),'\n')

#Question 1
#Is there any significant difference in sales in different regions?
sns.barplot(x = data['Region'],y=data['Sales'],ci=None)

#Question 2
#How are the states performing in terms of sales?
plt.figure(figsize=(15,15))
sns.barplot(x = data['Sales'],y=data['State'],ci=None)

#getting product insights
print(data['Category'].unique(),'\n')
print(data['Sub-Category'].unique(),'\n')

#Which product has the highest sales?
sns.barplot(x=data['Category'],y=data['Sales'],ci=None)

plt.figure(figsize=(15,10))
sns.barplot(x=data['Sub-Category'],y=data['Sales'],ci=None)

#Getting to know how discounts are doing for the business
sns.barplot(x=data['Discount'],y=data['Profit'],ci=None)

