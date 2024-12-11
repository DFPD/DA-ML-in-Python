"""
Assignment: Python Programming Midterm 1 Question 3 Part 2a
Author: Jason Zhao
Student ID: T12611201
Date created: April 9, 2024
"""


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv')
df.head(10)


# In[2]:


df['Sex_Code'] = df['Sex'].map({'female':1, 'male':0}).astype('int')
df['Sex'] = df['Sex_Code']
df['Age'] = df['Age'].fillna(df['Age'].mean())
df


# In[3]:


X = df[["Pclass", "Sex", "Age", "SibSp", "Parch"]]
# Adding SipSp and Parch as predictors
y = df["Survived"]


# In[4]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X,y)

print("Logistic Regression:")

# Changing predictor parametrs for Jack and Rose
Jack = clf.predict([[3, 0, 20.0, 0, 0]])
print("Jack", Jack)

# Fiance is ignored, so SibSp = 0
Rose = clf.predict([[1, 1, 17.0, 0, 1]])
print("Rose", Rose)
print()


# In[5]:


# K-NN
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier()
clf.fit(X,y)

# NEW parameters
print("K-NN:")
Jack = clf.predict([[3, 0, 20.0, 0, 0]])
print("Jack", Jack)
Rose = clf.predict([[1, 1, 17.0, 0, 1]])
print("Rose", Rose)
print()


# In[6]:


# SVC
from sklearn import svm
clf = svm.SVC()
clf.fit(X,y)

# NEW parameters
print("SVC:")
Jack = clf.predict([[3, 0, 20.0, 0, 0]])
print("Jack", Jack)
Rose = clf.predict([[1, 1, 17.0, 0, 1]])
print("Rose", Rose)
print()


# In[7]:


# Gaussian Naive bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X,y)

# NEW parameters
print("Guassian Naive Bayes")
Jack = clf.predict([[3, 0, 20.0, 0, 0]])
print("Jack", Jack)
Rose = clf.predict([[1, 1, 17.0, 0, 1]])
print("Rose", Rose)
print()


# In[8]:


# Multinomail Naive bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X,y)

# NEW parameters
print("Multinomial Naive Bayes:")
Jack = clf.predict([[3, 0, 20.0, 0, 0]])
print("Jack", Jack)
Rose = clf.predict([[1, 1, 17.0, 0, 1]])
print("Rose", Rose)
print()


# In[9]:


# Decision Tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X,y)

# NEW parameters
print("Decision Tree:")
Jack = clf.predict([[3, 0, 20.0, 0, 0]])
print("Jack", Jack)
Rose = clf.predict([[1, 1, 17.0, 0, 1]])
print("Rose", Rose)
print()

# In[10]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X,y)

# NEW parameters
print("Random Forest:")
Jack = clf.predict([[3, 0, 20.0, 0, 0]])
print("Jack", Jack)
Rose = clf.predict([[1, 1, 17.0, 0, 1]])
print("Rose", Rose)
print()

# In[11]:


# XGBoost
from xgboost.sklearn import XGBClassifier
clf = XGBClassifier()
clf.fit(X,y)
two = {"Pclass":[3,1], "Sex":[0,1], "Age":[20.0,17.0], "SibSp": [0,0], "Parch": [0,1]}
df1 = pd.DataFrame(two)

print("XGBoost:")
Jack = clf.predict(df1.iloc[0:1])
print("Jack", Jack)
Rose = clf.predict(df1.iloc[1:2])
print("Rose", Rose)
print()


