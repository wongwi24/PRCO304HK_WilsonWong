#!/usr/bin/env python
# coding: utf-8

# In[100]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[101]:


card_data = pd.read_csv('creditcard.csv\creditcard.csv')
print(card_data.shape)
X = card_data.iloc[:, :-1]
Y = card_data.iloc[:, -1]


# In[102]:


#Split dataset into test train and valid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify = Y, test_size = 0.25, random_state = 5)


# In[103]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[104]:


weight_nf = y_train.value_counts()[0] / len(y_train)
weight_f = y_train.value_counts()[1] / len(y_train)
print(f"Non-Fraud weight: {weight_nf}")
print(f"Fraud weight: {weight_f}")


# In[105]:


print(f"Train Data shape: {x_train.shape} Train Class Data shape: {y_train.shape}")
print(f"Test Data shape: {x_test.shape} Test Class Data shape: {y_test.shape}")


# In[106]:


from sklearn.decomposition import PCA 
pca = PCA(n_components = 27)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
var_explained = pca.explained_variance_ratio_.sum()
print(var_explained)


# In[107]:


ann = tf.keras.models.Sequential()


# In[108]:


ann.add(tf.keras.layers.Dense(units = 28, activation = "relu"))
ann.add(tf.keras.layers.Dense(units = 15, activation = "relu"))
ann.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid"))


# In[109]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[110]:


ann.fit(x_train, y_train, batch_size = 32, epochs = 100)


# In[111]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
def print_classification_result(true, predict):
    print(f"Accuracy Score: {accuracy_score(true, predict) * 100:.2f}%")
    print(f"Confusion Matrix: \n {confusion_matrix(true, predict)}\n")
    print(f"ROC_AUC_Score:{roc_auc_score(true, predict)}")


# In[112]:


y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)


# In[113]:


print_classification_result(y_test, y_pred)


# In[ ]:




