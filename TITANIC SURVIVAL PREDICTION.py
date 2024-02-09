# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:33:01 2023

@author: M GNANESHWARI
"""
"""
Business Problem Understanding
It is your job to predict if a passenger survived the sinking of the Titanic or not.
For each in the test set, you must predict a 0 or 1 value for the variable.
"""
"""
Performing Data Cleaning and Analysis
Understanding meaning of each column:
Data Dictionary: Variable Description

Survived - Survived (1) or died (0) Pclass - Passenger’s class (1 = 1st, 2 = 2nd, 3 = 3rd) Name - Passenger’s name Sex - Passenger’s sex Age - Passenger’s age SibSp - Number of siblings/spouses aboard Parch - Number of parents/children aboard (Some children travelled only with a nanny, therefore parch=0 for them.) Ticket - Ticket number Fare - Fare Cabin - Cabin Embarked - Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

Analysing which columns are completely useless in predicting the survival and deleting them
Note - Don't just delete the columns because you are not finding it useful. Or focus is not on deleting the columns. Our focus is on analysing how each column is affecting the result or the prediction and in accordance with that deciding whether to keep the column or to delete the column or fill the null values of the column by some values and if yes, then what values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings 
warnings.simplefilter("ignore")

titanic = pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\train.csv')
print(titanic.tail())
print(titanic)
print(titanic.columns)
"""
Data Understanding
survival = Survival (0 = No, 1 = Yes)

pclass = Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)

sex = Gender of Person

Age = Age of preson who travel in years

sibsp = of siblings / spouses aboard the Titanic (sibsp: The dataset defines family relations in this way...

                                                       Sibling = brother, sister, stepbrother, stepsister

                                                       Spouse = husband, wife (mistresses and fiancés were ignored)
parch = of parents / children aboard the Titanic (parch: The dataset defines family relations in this way...

                                                       Parent = mother, father

                                                       Child = daughter, son, stepdaughter, stepson)
ticket = Ticket number

fare = Passenger fare

cabin = Cabin number

embarked = Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

Dataset Understanding

Performing Data Cleaning and Analysis
Understanding meaning of each column: Data Dictionary: Variable Description
Survived - Survived (1) or died (0) Pclass - Passenger’s class (1 = 1st, 2 = 2nd, 3 = 3rd)
Name - Passenger’s name Sex - Passenger’s sex Age - Passenger’s age SibSp - Number of siblings/spouses aboard Parch - Number of parents/children aboard (Some children travelled only with a nanny, therefore parch=0 for them.) 
Ticket - Ticket number Fare - Fare Cabin - Cabin Embarked - Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

Analysing which columns are completely useless in predicting the survival and deleting them Note - Don't just delete the columns because you are not finding it useful. Or focus is not on deleting the columns. 
Our focus is on analysing how each column is affecting the result or the prediction and in accordance with that deciding whether to keep the column or to delete the column or fill the null values of the column by some values and if yes, then what values.
"""

print(titanic.describe())
#Name column can never decide survival of a person, hence we can safely delete it
del titanic["Name"]
titanic.head()
del titanic["Ticket"]
titanic.head()
del titanic["Fare"]
titanic.head()
del titanic['Cabin']
titanic.head()
titanic.info()
titanic.shape
"""
Data Preprocessing
EDA
"""

titanic["Survived"].value_counts()


titanic["Embarked"].value_counts()

titanic["Pclass"].value_counts()

titanic["SibSp"].value_counts()

titanic["Parch"].value_counts()

titanic.describe()
sns.countplot(data=titanic,x="Survived")
plt.xticks(ticks=[0,1],labels=["Not Survived","Survived"])
plt.title("Survived Count")
plt.show()
cor=titanic.corr()
cor
sns.heatmap(cor,annot=True)

plt.show()
titanic.drop(["PassengerId"],axis=1,inplace=True)
titanic
titanic.isnull().sum()
titanic.duplicated().sum()
titanic.drop_duplicates(inplace=True)
titanic["Embarked"].fillna(titanic["Embarked"].mode()[0],inplace=True)
titanic["Age"].fillna(titanic["Age"].mean(),inplace=True)



#Data Wrangling

titanic["Sex"].replace({"female":0,"male":1},inplace=True)
dum=pd.get_dummies(titanic["Embarked"],drop_first=True)
titanic=pd.concat([titanic,dum],axis="columns")
#titanic.drop("Embarked",axis=1,inplace=True)
titanic
#X & Y
x=titanic.drop("Survived",axis=1)
y=titanic["Survived"]
x.info()



# Changing Value for "Male, Female" string values to numeric values , male=1 and female=2
def getNumber(str):
    if str=="male":
        return 1
    else:
        return 2
#Pandas.apply allow the users to pass a function and apply it on every single value of the Pandas series. It comes as a huge improvement for the pandas library as this function helps to segregate data according to the conditions required due to which it is efficiently used in data science and machine learning.
      
titanic["Gender"]=titanic["Sex"].apply(getNumber)
#We have created a new column called "Gender" and 
#filling it with values 1,2 based on the values of sex column
print(titanic.head())
#Deleting Sex column, since no use of it now
del titanic["Sex"]
titanic.head()
titanic.isnull().sum()
titanic.columns

"""
Fill the null values of the Age column. Fill mean Survived age(mean age of the survived people) in the column where the person has survived and mean not Survived age (mean age of the people who have not survived) in the column where person has not survived
"""
meanS= titanic[titanic.Survived==1].Age.mean()
print(meanS)
"""
Creating a new "Age" column , filling values in it with a condition if goes True then given values (here meanS) is put in place of last values else nothing happens, simply the values are copied from the "Age" column of the dataset
"""
titanic["age"]=np.where(pd.isnull(titanic.Age) & titanic["Survived"]==1  ,meanS, titanic["Age"])
titanic.head()
titanic.isnull().sum()
# Finding the mean age of "Not Survived" people
meanNS=titanic[titanic.Survived==0].Age.mean()
meanNS
titanic.age.fillna(meanNS,inplace=True)
titanic.head()
titanic.isnull().sum()
del titanic['Age']
titanic.head()
"""
We want to check if "Embarked" column is is important for analysis or not, that is whether survival of the person depends on the Embarked column value or not
"""
# Finding the number of people who have survived 
# given that they have embarked or boarded from a particular port

survivedQ = titanic[titanic.Embarked == 'Q'][titanic.Survived == 1].shape[0]
survivedC = titanic[titanic.Embarked == 'C'][titanic.Survived == 1].shape[0]
survivedS = titanic[titanic.Embarked == 'S'][titanic.Survived == 1].shape[0]
print(survivedQ)
print(survivedC)
print(survivedS)
survivedQ = titanic[titanic.Embarked == 'Q'][titanic.Survived == 0].shape[0]
survivedC = titanic[titanic.Embarked == 'C'][titanic.Survived == 0].shape[0]
survivedS = titanic[titanic.Embarked == 'S'][titanic.Survived == 0].shape[0]
print(survivedQ)
print(survivedC)
print(survivedS)
"""
As there are significant changes in the survival rate based on which port the passengers aboard the ship. We cannot delete the whole embarked column(It is useful). Now the Embarked column has some null values in it and hence we can safely say that deleting some rows from total rows will not affect the result. So rather than trying to fill those null values with some vales. We can simply remove them.
"""
titanic.dropna(inplace=True)
titanic.head()
titanic.isnull().sum()
#Renaming "age" and "gender" columns
titanic.rename(columns={'age':'Age'}, inplace=True)
titanic.head()
titanic.rename(columns={'Gender':'Sex'}, inplace=True)
titanic.head()
def getEmb(str):
    if str=="S":
        return 1
    elif str=='Q':
        return 2
    else:
        return 3
titanic["Embark"]=titanic["Embarked"].apply(getEmb)
titanic.head()
del titanic['Embarked']
titanic.rename(columns={'Embark':'Embarked'}, inplace=True)
titanic.head()
#Drawing a pie chart for number of males and females aboard
import matplotlib.pyplot as plt
from matplotlib import style

males = (titanic['Sex'] == 1).sum() 

#Summing up all the values of column gender with a 
#condition for male and similary for females
females = (titanic['Sex'] == 2).sum()
print(males)
print(females)
p = [males, females]
plt.pie(p,    #giving array
       labels = ['Male', 'Female'], #Correspndingly giving labels
       colors = ['green', 'yellow'],   # Corresponding colors
       explode = (0.15, 0),    #How much the gap should me there between the pies
       startangle = 0)  #what start angle should be given
"""Matplotlib is a plotting library for creating static, animated, and interactive visualizations in Python. Pyplot is a Matplotlib module which provides a MATLAB-like interface. Matplotlib is designed to be as usable as MATLAB, with the ability to use Python and the advantage of being free and open-source.
"""
"""
matplotlib.pyplot.axis()
This function is used to set some axis properties to the graph.

Syntax: matplotlib.pyplot.axis(*args, emit=True, **kwargs)

Parameters:
xmin, xmax, ymin, ymax:These parameters can be used to
set the axis limits on the graph
emit:Its a bool value used to notify observers of the axis limit change"""

plt.axis('equal') 
plt.show()
# More Precise Pie Chart
MaleS=titanic[titanic.Sex==1][titanic.Survived==1].shape[0]
print(MaleS)
MaleN=titanic[titanic.Sex==1][titanic.Survived==0].shape[0]
print(MaleN)
FemaleS=titanic[titanic.Sex==2][titanic.Survived==1].shape[0]
print(FemaleS)
FemaleN=titanic[titanic.Sex==2][titanic.Survived==0].shape[0]
print(FemaleN)
chart=[MaleS,MaleN,FemaleS,FemaleN]
colors=['lightskyblue','yellowgreen','Yellow','Orange']
labels=["Survived Male","Not Survived Male","Survived Female","Not Survived Female"]
explode=[0,0.05,0,0.1]
plt.pie(chart,labels=labels,colors=colors,explode=explode,startangle=100,counterclock=False,autopct="%.2f%%")
"""
autopct: This parameter is a string or function used to label the wedges with their numeric value. 
colors: This parameter is the sequence of matplotlib color args through which the pie chart will cycle. label: This parameter is the sequence of strings providing the labels for each wedge.
"""
plt.axis("equal")
plt.show()
#x_test=pd.read_csv("/Users/anirbandutta/Desktop/Machine Learning/test_titanic_x_test.csv")
#x_test.head()
#same for x_test..
titanic = pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\train.csv')
print(titanic.tail())
print(titanic)
print(titanic.isnull())
print(MaleS)
titanic.dropna()
#SPLITING THE DATASET IN TRAINING SEfrom T & TESTING SET
# split the data to independent variable 
x=titanic.iloc[:,:-1].values
# split the data to dependent variabel 

y=titanic.iloc[:,1].values
# as d.v is continus that regression algorithm 
# as in the data set we have 2 attribute we slr algo

# split the dataset to 80-20%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.25, random_state = 0)
#x_test=pd.read_csv(r'C:\Users\M GNANESHWARI\Desktop\")
#x_test.head()
#we called simple linear regression algoriytm from sklearm framework 

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# we build simple linear regression model regressor
regressor.fit(x_train, y_train)
# test the model & create a predicted table 
y_pred = regressor.predict(x_test)
# visualize train data point ( 24 data)
plt.scatter(x_train, y_train, color = 'red') 
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visulaize test data point 
plt.scatter(x_test, y_test, color = 'red') 
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# slope is generrated from linear regress algorith which fit to dataset 
m = regressor.coef_

# interceppt also generatre by model. 
c = regressor.intercept_

# predict or forcast the future the data which we not trained before 
y_12 = 9312 * 12 + 26780
y_12

y_20 = 9312 * 20 + 26780
y_20


# to check overfitting  ( low bias high variance)
bias = regressor.score(x_train, y_train)
bias


# to check underfitting (high bias low variance)
variance = regressor.score(x_test,y_test)
variance

titanic["age"]=np.where(pd.isnull(titanic.Age) & titanic["Survived"]==1  ,meanS, titanic["Age"])
# deployment in flask & html 
# mlops (azur, googlcolab, heroku, kubarnate)

# Modeling & Evaluation by default Parameters
# Modeling
from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier(random_state=0)
dt_model.fit(x_train,y_train)

# Prediction
ytrain_pred=dt_model.predict(x_train)
ytest_pred=dt_model.predict(x_test)

# Evaluation
from sklearn.metrics import accuracy_score
print("Train:",accuracy_score(y_train,ytrain_pred))
print("Test:",accuracy_score(y_test,ytest_pred))

# CV
from sklearn.model_selection import cross_val_score
print("cross validation:",cross_val_score(dt_model,x,y,cv=5).mean())

#Visualization Of Tree
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10),dpi=100)   # Size of the tree which size we have
plot_tree(dt_model,
          filled=True,                # filled=True means it fills the color
          feature_names=x.columns,    # feature_name= it shows the name of the column is divided in decision node
          class_names=["0","1"])  # It shows the category of the each class 
plt.show()  

#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
estimator=DecisionTreeClassifier(random_state=0)
param_grid={"criterion":["gini","entropy"],
           "max_depth":list(range(1,20))}

grid_model=GridSearchCV(estimator,param_grid,cv=5,scoring="accuracy")
grid_model.fit(x_train,y_train)
grid_model.best_params_

hp_model=DecisionTreeClassifier(criterion='entropy', max_depth=5)
hp_model.fit(x_train,y_train)

hp_model.feature_importances_

features=pd.DataFrame(data=hp_model.feature_importances_,
                     index=x.columns,
                     columns=["Feature Importance"])
important_features=features[features["Feature Importance"]>0.01]
important_features

important_features_list=important_features.index.to_list()
important_features_list

#Final Model With Best Parameters & Important Features
# Input data with important features
x_new=x[important_features_list]

# Train-Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_new,y,test_size=0.2,random_state=12)

# Modeling
from sklearn.tree import DecisionTreeClassifier
final_dt_model=DecisionTreeClassifier(criterion="entropy",max_depth=5)
final_dt_model.fit(x_train,y_train)

# Prediction
ytrain_pred=final_dt_model.predict(x_train)
ytest_pred=final_dt_model.predict(x_test)


# Evaluation
from sklearn.metrics import accuracy_score
print("Train:",accuracy_score(y_train,ytrain_pred))
print("Test:",accuracy_score(y_test,ytest_pred))

# CV
from sklearn.model_selection import cross_val_score
print("cross validation:",cross_val_score(final_dt_model,x,y,cv=5).mean())

#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,ytest_pred)

#Classification Report 
from sklearn.metrics import classification_report
print(classification_report(y_test,ytest_pred))

#Save Model
test1=pd.read_csv(r"C:\Users\dell\Downloads\test.csv")
test1

"""
test = titanic.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)

test["Fare"].fillna(test["Fare"].mean(),inplace=True)
#test.drop_duplicates(inplace=True)
test.duplicated().sum()
test
test.isnull().sum()
test["Sex"].replace({"female":0,"male":1},inplace=True)
dum=pd.get_dummies(test["Embarked"],drop_first=True)
test_df=pd.concat([test,dum],axis="columns")

test_df.drop(["Embarked","Q","S","Parch"],axis=1,inplace=True)
test_df
output=final_dt_model.predict(test_df)
output

test_output=pd.DataFrame({"PassengerId":test1["PassengerId"],"Survived":output})
test_output


test_output.to_csv("titanic.csv",index=False)
"""
df=pd.read_csv("titanic.csv")
df
