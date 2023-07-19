#import libraries
import pandas as pd

#read dataset
Adult_income=pd.read_csv("AdultIncome.csv")

#check for Null Values
Nan_val=Adult_income.isnull().sum(axis=0)

#get Dummy variable
data_prep=pd.get_dummies(Adult_income,drop_first=True)

#Create X and Y variable
X=data_prep.iloc[:,:-1]
Y=data_prep.iloc[:,-1]

#split the X and Y dataset into training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3, random_state=1234,stratify=Y)

#import and train classifier
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(random_state=1234)
dtc.fit(x_train,y_train)

#test Model
Y_predict=dtc.predict(x_test)
Y_validate=dtc.predict(x_train)
#model evaluation
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, Y_predict)

cm_validate=confusion_matrix(y_train,Y_validate)
score =dtc.score(x_test, y_test)
score_validate=dtc.score(x_train, y_train)
print(f"Confusion metrix for adult income classifier:\n {cm}")
print(f"accuracy score of adult income classifier: {score}")

print(f"Confusion metrix(valideation) for adult income classifier:\n {cm_validate}")
print(f"accuracy validateion score of adult income classifier: {score_validate}")