# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.Calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Haridharshini J
RegisterNumber:  212224040098
*/


import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:
## Data head:
<img width="354" height="273" alt="Screenshot 2025-10-13 094230" src="https://github.com/user-attachments/assets/7830f71f-9ee3-4a78-90aa-22104a237c0c" />
## Data info:
<img width="748" height="289" alt="Screenshot 2025-10-13 094247" src="https://github.com/user-attachments/assets/7c748211-111a-43f2-a39a-f79109cff507" />
## isnull() sum()
<img width="199" height="116" alt="Screenshot 2025-10-13 094304" src="https://github.com/user-attachments/assets/dc0abd48-88c5-4242-b107-460eae300744" />
## Mean squared error:
<img width="237" height="51" alt="Screenshot 2025-10-13 094341" src="https://github.com/user-attachments/assets/033af8db-109b-48a1-ae63-1f1cb0a9cd9e" />
## r2 value:
<img width="223" height="43" alt="Screenshot 2025-10-13 094422" src="https://github.com/user-attachments/assets/1b31cd09-25d3-495c-afad-779f84d19b2d" />
## data prediction
<img width="213" height="42" alt="Screenshot 2025-10-13 094506" src="https://github.com/user-attachments/assets/0638e79b-0810-4e7f-b744-f9d92845024e" />





## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
