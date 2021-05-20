#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer


# In[4]:


card_data = pd.read_csv('creditcard.csv\creditcard.csv')
print(card_data.shape)
X = card_data.iloc[:, :-1]
Y = card_data.iloc[:, -1]


# ## Data Preprocessing

# In[5]:


#Split dataset into test train and valid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify = Y, test_size = 0.25, random_state = 5)


# In[6]:


ct = ColumnTransformer([
        ('std', StandardScaler(), ['Amount', 'Time'])
    ], remainder='passthrough')
x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)


# In[7]:


weight_nf = y_train.value_counts()[0] / len(y_train)
weight_f = y_train.value_counts()[1] / len(y_train)
print(f"Non-Fraud weight: {weight_nf}")
print(f"Fraud weight: {weight_f}")


# In[8]:


print(f"Train Data shape: {x_train.shape} Train Class Data shape: {y_train.shape}")
print(f"Test Data shape: {x_test.shape} Test Class Data shape: {y_test.shape}")


# In[9]:


from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_score, matthews_corrcoef


def print_classification_result(true, predict):
    print(f"Accuracy Score: {accuracy_score(true, predict) * 100:.2f}%")
    print(f"Confusion Matrix: \n {confusion_matrix(true, predict)}\n")
    print(f"MCC_Score:{matthews_corrcoef(true, predict)}\n")
    print(f"f1_score: \n {f1_score(true, predict)}\n")
    print(f"recall_score: \n {recall_score(true, predict)}\n")
    print(f"Precision_Score:{precision_score(true, predict)}")


# In[43]:


x_train = np.array(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = np.array(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)


# ## Convolution Neural Network

# In[74]:


cnn = tf.keras.models.Sequential()


# In[75]:


cnn.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 3, activation = 'relu', input_shape = [30, 1]))
cnn.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 1))


# In[76]:


cnn.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 1))


# In[77]:


cnn.add(tf.keras.layers.Flatten())


# In[78]:


cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))


# In[79]:


cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# In[80]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[81]:


loss = cnn.fit(x_train, y_train, batch_size = 32, epochs = 10, verbose = 1, validation_split=0.3)


# In[82]:


plt.plot(loss.history['loss'])
plt.plot(loss.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[83]:


y_pred = cnn.predict(x_test)
y_pred = np.round(y_pred)
print_classification_result(y_test, y_pred)


# ## Parameter Testing

# In[31]:


cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 4, activation = 'relu', input_shape = [x_train.shape[1], 1]))
cnn.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 1))
cnn.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 1))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[32]:


loss = cnn.fit(x_train, y_train, batch_size = 32, epochs = 10, verbose = 1, validation_split = 0.3)


# In[33]:


plt.plot(loss.history['loss'])
plt.plot(loss.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[34]:


y_pred = cnn.predict(x_test)
y_pred = np.round(y_pred)
print_classification_result(y_test, y_pred)


# In[35]:


cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv1D(filters = 128, kernel_size = 3, activation = 'relu', input_shape = [x_train.shape[1], 1]))
cnn.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 1))
cnn.add(tf.keras.layers.Conv1D(filters = 128, kernel_size = 2, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 1))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[36]:


loss = cnn.fit(x_train, y_train, batch_size = 32, epochs = 10, verbose = 1, validation_split = 0.3)


# In[37]:


plt.plot(loss.history['loss'])
plt.plot(loss.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[38]:


y_pred = cnn.predict(x_test)
y_pred = np.round(y_pred)
print_classification_result(y_test, y_pred)


# In[39]:


cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [x_train.shape[1], 1]))
cnn.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 1))
cnn.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 2, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 1))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[40]:


loss = cnn.fit(x_train, y_train, batch_size = 32, epochs = 10, verbose = 1, validation_split = 0.3)


# In[41]:


plt.plot(loss.history['loss'])
plt.plot(loss.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# In[42]:


y_pred = cnn.predict(x_test)
y_pred = np.round(y_pred)
print_classification_result(y_test, y_pred)


# In[ ]:




