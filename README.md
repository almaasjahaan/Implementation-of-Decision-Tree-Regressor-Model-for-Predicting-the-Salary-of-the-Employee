# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.Calculate Mean square error,data prediction and r2.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: ALMAAS JAHAAN M
RegisterNumber: 212224230016
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:\\Users\\admin\\Downloads\\Salary.csv")

print(data.head())          # View first 5 rows
print(data.info())          # Dataset info
print(data.isnull().sum())  # Check for null values

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())  # View updated dataset

x = data[["Position", "Level"]]  # Features
y = data["Salary"]               # Target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)

dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)

print("Predicted Salary for [5,6]:", dt.predict([[5, 6]]))

plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()

```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
![image](https://github.com/user-attachments/assets/79d5f548-f945-4aa7-83a4-92c10dea55e1)
![image](https://github.com/user-attachments/assets/9e93bed9-49a5-4c92-b133-aead37488952)
![image](https://github.com/user-attachments/assets/8c2a3479-f8c3-49ee-9cc6-22f1ea9be8cc)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
