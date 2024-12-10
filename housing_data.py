#!/usr/bin/env python
# coding: utf-8

# # housing data

# I'll choose regression to be the prediction model because it's impossible/hard to set 1000 classifications/subsets from 1000 data. 

# In[1]:


import numpy as np
import pandas as pd

df = pd.read_csv("housing_data.csv")
df


# In[2]:


# Translate text format to symbol(int) format
df["Location"] = df["Location"].map({"City Center":0, "Suburb":1, "Rural":2}).astype('int')
df


# # Sell Price Prediction for example

# In[3]:


# Select X and y (sell price)
X = df.iloc[:,0:5]
y = df.iloc[:,6]


# In[4]:


# Create predict dataset
predictdata = {"Area":[3000,100,3000], "No. of Rooms":[2,1,4], "No. of Bathrooms":[1,1,2], "Location":[0,1,2], "Miles (dist. between school and house)":[10,50,350]}
predictdatadf = pd.DataFrame(predictdata)
predictdatadf


# In[5]:


# Linear
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y)
print(reg.predict(predictdatadf))


# # Idea: Find similar data by house 1's features

# In[6]:


df[df["Area"] == 3000]


# In[7]:


df[df["No. of Rooms"] == 2]


# In[8]:


df[df["No. of Bathrooms"] == 1]


# In[9]:


df[df["Location"] == 0]


# In[10]:


df[df["Miles (dist. between school and house)"] == 10]


# According to above filters, I think the predicted house 1 price ... (please finish the story of house 1. Similar to houses 2 and 3.)

# In[11]:


# You may try polynomial regression
# Polynomial power = 2
from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree = 2)
X_poly = poly_reg2.fit_transform(X)
regp2 = LinearRegression()
regp2.fit(X_poly, y)


predictdatadf_poly = poly_reg2.fit_transform(predictdatadf)
print(regp2.predict(predictdatadf_poly))


# In[12]:


# Polynomial power = 3
from sklearn.preprocessing import PolynomialFeatures
poly_reg3 = PolynomialFeatures(degree = 3)
X_poly = poly_reg3.fit_transform(X)
regp3 = LinearRegression()
regp3.fit(X_poly, y)


predictdatadf_poly = poly_reg3.fit_transform(predictdatadf)
print(regp3.predict(predictdatadf_poly))


# In[ ]:




