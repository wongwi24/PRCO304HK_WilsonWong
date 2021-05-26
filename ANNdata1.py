#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt


# In[2]:


card_data = pd.read_csv('creditcard.csv\creditcard.csv')
print(card_data.shape)
X = card_data.iloc[:, :-1]
Y = card_data.iloc[:, -1]


# ## Data Preprocessing

# In[3]:


#Split dataset into test train and valid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify = Y, test_size = 0.25, random_state = 5)


# In[4]:


ct = ColumnTransformer([
        ('std', StandardScaler(), ['Amount', 'Time'])
    ], remainder='passthrough')
x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)


# In[5]:


weight_nf = y_train.value_counts()[0] / len(y_train)
weight_f = y_train.value_counts()[1] / len(y_train)
print(f"Non-Fraud weight: {weight_nf}")
print(f"Fraud weight: {weight_f}")


# In[6]:


print(f"Train Data shape: {x_train.shape} Train Class Data shape: {y_train.shape}")
print(f"Test Data shape: {x_test.shape} Test Class Data shape: {y_test.shape}")


# ## Artificial Neural Network

# In[7]:


ann = tf.keras.models.Sequential()


# In[8]:


ann.add(tf.keras.layers.Dense(units = 31, activation = "relu"))
ann.add(tf.keras.layers.Dense(units = 15, activation = "relu"))
ann.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid"))


# In[9]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[10]:


loss = ann.fit(x_train, y_train, batch_size = 32, epochs = 11, validation_split = 0.3)


# In[7]:


from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_score, matthews_corrcoef


def print_classification_result(true, predict):
    print(f"Accuracy Score: {accuracy_score(true, predict) * 100:.2f}%")
    print(f"Confusion Matrix: \n {confusion_matrix(true, predict)}\n")
    print(f"MCC_Score:{matthews_corrcoef(true, predict)}\n")
    print(f"f1_score: \n {f1_score(true, predict)}\n")
    print(f"recall_score: \n {recall_score(true, predict)}\n")
    print(f"Precision_Score:{precision_score(true, predict)}")


# In[12]:


plt.plot(loss.history['loss'])
plt.plot(loss.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[13]:


y_pred = ann.predict(x_test)
y_pred = np.round(y_pred)


# In[14]:


print_classification_result(y_test, y_pred)


# ## Parameter testing

# In[8]:


ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 31, activation = "relu"))
ann.add(tf.keras.layers.Dense(units = 20, activation = "relu"))
ann.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid"))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[9]:


loss = ann.fit(x_train, y_train, batch_size = 32, epochs = 11, validation_split = 0.3)


# In[10]:


plt.plot(loss.history['loss'])
plt.plot(loss.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[11]:


y_pred = ann.predict(x_test)
y_pred = np.round(y_pred)


# In[12]:


print_classification_result(y_test, y_pred)


# In[13]:


ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 31, activation = "relu"))
ann.add(tf.keras.layers.Dense(units = 10, activation = "relu"))
ann.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid"))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[14]:


loss = ann.fit(x_train, y_train, batch_size = 32, epochs = 11, validation_split = 0.3)


# In[15]:


plt.plot(loss.history['loss'])
plt.plot(loss.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[17]:


y_pred = ann.predict(x_test)
y_pred = np.round(y_pred)


# In[18]:


print_classification_result(y_test, y_pred)


# In[ ]:




