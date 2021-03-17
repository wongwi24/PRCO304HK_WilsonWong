#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# ## Import and Analysis of data

# In[2]:


card_data = pd.read_csv('creditcard.csv\creditcard.csv')
print(card_data.shape)
X = card_data.iloc[:, :-1]
Y = card_data.iloc[:, -1]


# In[3]:


card_data.describe()


# In[4]:


# Check if there is null values
card_data.isnull().sum()


# In[5]:


#Plot Fraud vs Not Fraud transaction counts
fraud_count = card_data["Class"].value_counts()
plt.figure()
plt.title("Fraud Class Counts")
plt.bar(fraud_count.index,fraud_count.values,color='purple')
plt.xticks([0,1], labels=["not fraud","fraud"])
plt.show()


# In[6]:


#Plot features correlation
plt.figure(figsize = (13,13))
corr = card_data.corr()
plt.title("Features Correlation")
sb.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.show


# In[7]:


#Transaction Time density plot
notFraud_time = card_data.loc[card_data["Class"]==0]
fraud_time = card_data.loc[card_data["Class"]==1]
plt.figure()
sb.kdeplot(data = notFraud_time["Time"], label = "Class 0")
sb.kdeplot(data = fraud_time["Time"], label = "Class 1")
plt.legend()
plt.show()


# In[8]:


sb.boxplot(x = "Class", y = "Amount", hue = "Class", data = card_data, showfliers = False)
plt.show()


# In[9]:


#Features density plot
col = card_data.columns.values
c0 = card_data.loc[card_data["Class"] == 0]
c1 = card_data.loc[card_data["Class"] == 1]
x = 0
plt.figure()
plt.subplots(8,4,figsize=(17,27))

for feature in col:
    x += 1
    plt.subplot(8,4,x)
    sb.kdeplot(data = c0[feature], label = "X Fraud")
    sb.kdeplot(data = c1[feature], label = "Fraud")
    plt.xlabel(feature, fontsize=10)
    plt.legend()
plt.show()


# ## Data Preprocessing

# In[10]:


#Split dataset into test train and valid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train_v, x_test, y_train_v, y_test = train_test_split(X, Y, stratify = Y, test_size = 0.25, random_state = 5)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_v, y_train_v, stratify = y_train_v, test_size = 0.25, random_state = 5)


# In[11]:


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
x_valid = sc.fit_transform(x_valid)


# In[12]:


weight_nf = y_train.value_counts()[0] / len(y_train)
weight_f = y_train.value_counts()[1] / len(y_train)
print(f"Non-Fraud weight: {weight_nf}")
print(f"Fraud weight: {weight_f}")


# In[13]:


print(f"Train Data shape: {x_train.shape} Train Class Data shape: {y_train.shape}")
print(f"Test Data shape: {x_test.shape} Test Class Data shape: {y_test.shape}")
print(f"Valid Data shape: {x_valid.shape} Valid Class Data shape: {y_valid.shape}")


# ## Random Forest Classifier

# In[14]:


from sklearn.ensemble import RandomForestClassifier
randfclassifier = RandomForestClassifier(n_estimators = 100, criterion = "entropy")
randfclassifier.fit(x_train, y_train)
y_valid_pred_randf = randfclassifier.predict(x_valid)


# In[15]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
def print_classification_result(true, predict):
    print(f"Accuracy Score: {accuracy_score(true, predict) * 100:.2f}%")
    print(f"Confusion Matrix: \n {confusion_matrix(true, predict)}\n")
    print(f"ROC_AUC_Score:{roc_auc_score(true, predict)}")


# In[16]:


print_classification_result(y_valid, y_valid_pred_randf)


# In[17]:


#Change number of trees in forest to 200
randfclassifier2 = RandomForestClassifier(n_estimators = 200, criterion = "entropy")
randfclassifier2.fit(x_train, y_train)
y_valid_pred_randf = randfclassifier2.predict(x_valid)
print_classification_result(y_valid, y_valid_pred_randf)


# In[18]:


y_test_pred_randf = randfclassifier2.predict(x_test)
print_classification_result(y_test, y_test_pred_randf)


# ## Kernel Support Vector Machine

# In[19]:


from sklearn.svm import SVC
KSVM = SVC(kernel = "rbf")
KSVM.fit(x_train, y_train)
y_test_pred_KSVM = KSVM.predict(x_test)
print_classification_result(y_test, y_test_pred_KSVM)


# ## K Nearest Neighbors

# In[20]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 6, p = 2, metric = "minkowski")
knn.fit(x_train, y_train)
y_valid_pred_KNN = knn.predict(x_valid)
print_classification_result(y_valid, y_valid_pred_KNN)


# In[21]:


#Change number of neighbors to 5
knn2 = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = "minkowski")
knn2.fit(x_train, y_train)
y_valid_pred_KNN = knn2.predict(x_valid)
print_classification_result(y_valid, y_valid_pred_KNN)


# In[22]:


y_test_pred_KNN = knn2.predict(x_test)
print_classification_result(y_test, y_test_pred_KNN)


# ## Naive Bayes Classifier

# In[23]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
y_test_pred_NB = nb.predict(x_test)
print_classification_result(y_test, y_test_pred_NB)


# In[ ]:




