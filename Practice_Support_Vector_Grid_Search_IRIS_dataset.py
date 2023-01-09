#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machines Project 
# 
# Welcome to your Support Vector Machine Project! Just follow along with the notebook and instructions below. We will be analyzing the famous iris data set!
# 
# **The Data**
# For this series of lectures, we will be using the famous Iris flower data set.
# 
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis.
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
# 
# Here's a picture of the three different Iris types:

# In[1]:


# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# In[2]:


# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)


# In[3]:


# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


# The iris dataset contains measurements for 150 iris flowers from three different species.
# 
# The three classes in the Iris dataset:
# 
# Iris-setosa (n=50)
# Iris-versicolor (n=50)
# Iris-virginica (n=50)
# The four features of the Iris dataset:
# 
# sepal length in cm
# sepal width in cm
# petal length in cm
# petal width in cm
# 
# Get the data
# 
# *Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') *

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import seaborn as sns
iris = sns.load_dataset('iris')


# In[5]:


iris


# In[7]:


species_dummies = pd.get_dummies(iris['species'])


# In[8]:


species_dummies


# In[14]:


sns.countplot(x='setosa', data=species_dummies)


# In[20]:


sns.pairplot(iris, hue='species',palette='Dark2')


# # Train Test Split
# 
# ** Split your data into a training set and a testing set.**

# In[22]:


from sklearn.model_selection import train_test_split
X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[23]:


from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)


# In[24]:


y_pred = svm.predict(X_test)


# In[25]:


y_pred


# In[26]:


from sklearn.metrics import classification_report, confusion_matrix


# In[27]:


print(classification_report(y_test, y_pred))


# In[28]:


print(confusion_matrix(y_test, y_pred))


# **The model is pretty good in it self. We may stop here but since its about the practice, let's see what we can achieve with the GridSearchCV here**

# In[29]:


from sklearn.model_selection import GridSearchCV


# In[32]:


param_grid = {
    'C' : [0.1, 1, 10, 100, 1000],
    'gamma' : [1, 0.1, 0.01, 0.001, 0.0001]
}


# In[33]:


grid_cv = GridSearchCV(SVC(), param_grid, verbose=2)


# In[34]:


grid_cv.fit(X_train, y_train)


# In[35]:


grid_cv.best_estimator_


# In[36]:


grid_cv.best_params_


# In[37]:


grid_cv.best_score_


# In[38]:


y_pred = grid_cv.predict(X_test)


# In[40]:


print(classification_report(y_test, y_pred))


# In[41]:


print(confusion_matrix(y_test, y_pred))


# **Pretty Good Results!**

# # Good Job
