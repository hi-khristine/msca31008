#!/usr/bin/env python
# coding: utf-8

# ## UFC Fight-level dataset SVM Notebook
# (thre is no quick rule as to which kernel performs best in every scenario; testing & learning is key)
# 
# Kernel trick reference:
# https://towardsdatascience.com/understanding-support-vector-machine-part-2-kernel-trick-mercers-theorem-e1e6848c6c4d

# #### Import necessary modules

# In[32]:


import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Helper functions

# In[50]:


# rtns best params for C and Gamma; they are the parameters for a nonlinear support vector machine

def svc_parameter_optimization(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10] # C is trade off betw. low train error and low test error (ability to generalize) 
    gammas = [0.001, 0.01, 0.1, 1] # free parameter of the Gaussian radial basis function
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds) # instantiate grid search
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# #### Set working directory

# In[2]:


os.chdir(r'/Users/colella2/Google Drive/Graduate School/MScA/Courses/31008 Data Mining Principles/Final_Project/msca31008/fun/')


# In[3]:


pwd


# In[4]:


os.listdir('../fun/') # confirm items in function folder


# In[5]:


exec(open('r.py').read()) # test ability to read .py script from function folder


# In[6]:


# read-in all the functions
for filename in os.listdir('../fun/'):
    if filename.endswith('.py'):
        exec(open(filename).read())
        continue
    else:
        continue


# #### Open file of interest

# In[7]:


os.chdir(r'/Users/colella2/Google Drive/Graduate School/MScA/Courses/31008 Data Mining Principles/Final_Project/msca31008/out')

with open('d3-fight-level-transform.pkl', 'rb') as f:
    data = pickle.load(f)


# In[8]:


load( '../out/d3-fight-level-transform.pkl' )
print( X.shape )


# #### Examine key-value pairs in dict

# In[9]:


for key, value in data.items():
  print(key, value)


# In[10]:


print(X.shape) # view feature shape; 4368 rows, 165 columns


# In[11]:


print(y.shape) # view predicted value shape; 4368 rows, 1 column


# #### Train-test split

# In[12]:


X_train , X_test, y_train, y_test = train_test_split(X, y, random_state = 718, test_size = 0.3)


# #### Fit model (linear kernel)
# (find decision boundary for linearly separable data)

# In[13]:


svclassifier_linear = SVC(kernel='linear')
svclassifier_linear.fit(X_train, y_train)


# In[14]:


# accuracy against train data
print(classification_report(y_train,svclassifier_linear.predict(X_train)))


# In[15]:


# accuracy against test data
print(classification_report(y_test, svclassifier_linear.predict(X_test)))


# In[16]:


# store predicted values on X_test & print confusion matrix
y_pred_linear = svclassifier_linear.predict(X_test)
print(confusion_matrix(y_test,y_pred_linear))


# ### Begin section for non-linear investigation

# #### Fit model (poly kernel)

# In[17]:


svclassifier_poly = SVC(kernel='poly', degree=8)
svclassifier_poly.fit(X_train, y_train)


# In[18]:


# accuracy against train data
print(classification_report(y_train,svclassifier_poly.predict(X_train)))


# In[19]:


# accuracy against test data
print(classification_report(y_test, svclassifier_poly.predict(X_test)))


# In[20]:


# store predicted values on X_test & print confusion matrix
y_pred_poly = svclassifier_poly.predict(X_test)
print(confusion_matrix(y_test,y_pred_poly))


# #### Fit model (Gaussian kernel)
# (this is a special case for rbf)

# In[21]:


svclassifier_gaus = SVC(kernel='rbf')
svclassifier_gaus.fit(X_train, y_train)


# In[22]:


# accuracy against train data
print(classification_report(y_train,svclassifier_gaus.predict(X_train)))


# In[23]:


# accuracy against test data
print(classification_report(y_test, svclassifier_gaus.predict(X_test)))


# In[24]:


# store predicted values on X_test & print confusion matrix
y_pred_gaus = svclassifier_gaus.predict(X_test)
print(confusion_matrix(y_test,y_pred_gaus))


# #### After first running simply (as was done above), commence optimization...

# In[36]:


# commence grid search for best parameters on training set
svc_parameter_optimization(X_train, y_train, 5)


# In[37]:


# instantiate with optimal parameters
svclassifier_gaus_optim = SVC(kernel='rbf', C = 1, gamma = 1)
svclassifier_gaus_optim.fit(X_train, y_train)


# In[38]:


# view optimized results/accuracy on training data
print(classification_report(y_train,svclassifier_gaus_optim.predict(X_train)))


# In[39]:


# view optimized results/accuracy on testing data
print(classification_report(y_test, svclassifier_gaus_optim.predict(X_test)))


# In[40]:


# store predicted values on X_test & print confusion matrix
y_pred_gaus_optim = svclassifier_gaus_optim.predict(X_test)
print(confusion_matrix(y_test,y_pred_gaus_optim))


# #### Fit model (Sigmoid kernel)
# (suitable for binary classification problems; rtns 0 or 1; activation functino for Neural Networks)

# In[25]:


svclassifier_sig = SVC(kernel='sigmoid')
svclassifier_sig.fit(X_train, y_train)


# In[26]:


# accuracy against train data
print(classification_report(y_train,svclassifier_sig.predict(X_train)))


# In[27]:


# accuracy against test data
print(classification_report(y_test, svclassifier_sig.predict(X_test)))


# In[28]:


# store predicted values on X_test & print confusion matrix
y_pred_sig = svclassifier_sig.predict(X_test)
print(confusion_matrix(y_test,y_pred_sig))


# #### Conclusion
# Linear SVM performs similarly to optimized Gaussian RBF on test sets.
