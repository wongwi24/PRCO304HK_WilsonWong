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
x = card_data.iloc[:, :-1]
y = card_data.iloc[:, -1]


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


# In[51]:


#Transaction Time density plot
notFraud_time = card_data.loc[card_data["Class"]==0]
fraud_time = card_data.loc[card_data["Class"]==1]
plt.figure()
sb.kdeplot(data = notFraud_time["Time"], label = "Class 0")
sb.kdeplot(data = fraud_time["Time"], label = "Class 1")
plt.legend()
plt.show()


# In[47]:


sb.boxplot(x = "Class", y = "Amount", hue = "Class", data = card_data, showfliers = False)
plt.show()


# In[66]:


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


# In[ ]:




