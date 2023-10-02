# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: JEEVITHA S
RegisterNumber:  212222100016


import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:

 ### Placement Data: 
 ![233679600-d7637871-ac7e-4ef8-8538-cfe8f8c1ddb3](https://github.com/Jeevithha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123623197/27accfb2-61bc-4490-a746-3abe87352eb6)

 ### Salary Data: 

 ![233679823-32ae13cc-489d-436a-925a-6187d6de27ed](https://github.com/Jeevithha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123623197/72d28d73-a35d-47b9-9b28-69048e18d580)

 ### Checking the null() function:

 ![233679969-7a2b5524-270d-4377-9728-f78188177f6c](https://github.com/Jeevithha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123623197/4a0678c0-37f5-4db4-b37a-5c9ab8079289)

### Data Duplicate:
![233680057-efb79829-4a73-4fab-9e37-01f58b54898b](https://github.com/Jeevithha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123623197/29243910-c697-4c6b-b7fe-135be3ed9f5e)

### Print Data: 

![233680198-69570dd4-1cce-4363-bce2-a4cae93e236e](https://github.com/Jeevithha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123623197/4a241307-5374-4125-a6e7-ddcdaf20d868)

### Data-status:

![233680590-861937d3-aba8-400c-8ccf-80c25444cd69](https://github.com/Jeevithha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123623197/63baa2d1-975b-48ff-b56a-d5328b81c334)

### y_prediction array:

![233680712-229c768c-f1c1-4ec8-b43f-0b0d2996ee31](https://github.com/Jeevithha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123623197/637679aa-2c30-4e8e-a78a-ffd6ce0ece0d)

### Accuracy value:

![233680788-7cbdbe90-d08b-4076-aac7-50a4ad6c26b0](https://github.com/Jeevithha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123623197/ed3a37d9-918c-4eb1-baa0-223b031aa189)

### Confusion array:

![233681332-f1ee5ca5-9812-40b9-8d7b-c3cda844fec3](https://github.com/Jeevithha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123623197/8ffeeb7f-a387-4e3f-af94-1eb74e0df912)

### Classification report:

![233681332-f1ee5ca5-9812-40b9-8d7b-c3cda844fec3](https://github.com/Jeevithha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123623197/ad97c04f-d643-4ecc-b4ca-ba1a7c30d8fe)

### Prediction of LR:

![233681412-e62e2859-e43f-4515-8a18-ae7ea8bc19cb](https://github.com/Jeevithha/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/123623197/603a63b5-858e-4244-8154-4c465c9e9db9)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
