#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
train_id = pd.read_csv('creditcard_dataset2.csv/train_identity.csv')
train_trans = pd.read_csv('creditcard_dataset2.csv/train_transaction.csv')


# ## Analysis of Data

# In[67]:


print(train_id.shape)
print(train_trans.shape)


# In[68]:


plt.figure(figsize=(10, 4))
sb.kdeplot(train_trans[train_trans['isFraud']==1]['TransactionDT'], label='Fraud', color = 'blue');
sb.kdeplot(train_trans[train_trans['isFraud']==0]['TransactionDT'], label='X Fraud', color = 'tab:orange');
plt.xlabel("TransactionDT", fontsize=10)
plt.legend()
plt.show()


# In[69]:


fig, ax = plt.subplots(1, 2, figsize=(15,4))

time_value = train_trans.loc[train_trans['isFraud'] == 1]['TransactionAmt'].values

sb.distplot(np.log(time_value), ax=ax[0], color='green')
ax[0].set_title('Distribution of TransactionAmt is Fraud', fontsize=14)
ax[0].set_xlim([min(np.log(time_value)), max(np.log(time_value))])

time_value = train_trans.loc[train_trans['isFraud'] == 0]['TransactionAmt'].values

sb.distplot(np.log(time_value), ax=ax[1], color='tab:purple')
ax[1].set_title('Distribution of TransactionAmt non Fraud', fontsize=14)
ax[1].set_xlim([min(np.log(time_value)), max(np.log(time_value))])


plt.show()


# In[70]:


plt.figure(figsize = (15, 4))
sb.countplot(x="ProductCD", hue = "isFraud", data=train_trans).set_title('ProductCD', fontsize=14)
plt.show()


# In[71]:


col = ['card1', 'card2', 'card3']
c0 = train_trans.loc[train_trans["isFraud"] == 0]
c1 = train_trans.loc[train_trans["isFraud"] == 1]
x = 0
plt.figure()
plt.subplots(1,3,figsize=(15,5))

for feature in col:
    x += 1
    plt.subplot(1,3,x)
    sb.kdeplot(data = c0[feature], label = "X Fraud")
    sb.kdeplot(data = c1[feature], label = "Fraud")
    plt.xlabel(feature, fontsize=10)
    plt.legend()
plt.show()


# In[72]:


fig, ax = plt.subplots(1, 2, figsize=(15,5))

sb.countplot(x="card4", ax=ax[0], data=train_trans.loc[train_trans['isFraud'] == 0])
ax[0].set_title('card4 is not Fraud', fontsize=14)
sb.countplot(x="card4", ax=ax[1], data=train_trans.loc[train_trans['isFraud'] == 1])
ax[1].set_title('card4 is Fraud', fontsize=14)


# In[73]:


plt.figure(figsize=(10, 4))
sb.kdeplot(train_trans[train_trans['isFraud']==1]['card5'], label='Fraud', color = 'blue');
sb.kdeplot(train_trans[train_trans['isFraud']==0]['card5'], label='X Fraud', color = 'tab:orange');
plt.xlabel("card5", fontsize=10)
plt.legend()
plt.show()


# In[74]:


fig, ax = plt.subplots(1, 2, figsize=(15,5))

sb.countplot(x="card6", ax=ax[0], data=train_trans.loc[train_trans['isFraud'] == 0])
ax[0].set_title('card6 non Fraud', fontsize=14)
sb.countplot(x="card6", ax=ax[1], data=train_trans.loc[train_trans['isFraud'] == 1])
ax[1].set_title('card6 is Fraud', fontsize=14)


# In[75]:


col = ['addr1', 'addr2']
c0 = train_trans.loc[train_trans["isFraud"] == 0]
c1 = train_trans.loc[train_trans["isFraud"] == 1]
x = 0
plt.figure()
plt.subplots(1,2,figsize=(15,5))

for feature in col:
    x += 1
    plt.subplot(1,2,x)
    sb.kdeplot(data = c0[feature], label = "X Fraud")
    sb.kdeplot(data = c1[feature], label = "Fraud")
    plt.xlabel(feature, fontsize=10)
    plt.legend()
plt.show()


# In[76]:


train_trans.loc[train_trans['P_emaildomain'].isin(['gmail.com', 'gmail']),'P_emaildomain'] = 'Google'

train_trans.loc[train_trans['P_emaildomain'].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk',
                                         'yahoo.co.jp', 'yahoo.de', 'yahoo.fr',
                                         'yahoo.es']), 'P_emaildomain'] = 'Yahoo Mail'
train_trans.loc[train_trans['P_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 
                                         'hotmail.es','hotmail.co.uk', 'hotmail.de',
                                         'outlook.es', 'live.com', 'live.fr',
                                         'hotmail.fr']), 'P_emaildomain'] = 'Microsoft'
train_trans.loc[train_trans.P_emaildomain.isin(train_trans.P_emaildomain                                         .value_counts()[train_trans.P_emaildomain.value_counts() <= 500 ]                                         .index), 'P_emaildomain'] = "Others"
train_trans.P_emaildomain.fillna("NoInf", inplace=True)

fig, ax = plt.subplots(2, 1, figsize=(24,12))

sb.countplot(x="P_emaildomain", ax=ax[0], data=train_trans.loc[train_trans['isFraud'] == 0])
ax[0].set_title('P_emaildomain is not Fraud', fontsize=14)
sb.countplot(x="P_emaildomain", ax=ax[1], data=train_trans.loc[train_trans['isFraud'] == 1])
ax[1].set_title('P_emaildomain is Fraud', fontsize=14)


# In[77]:


train_trans.loc[train_trans.C1.isin(train_trans.C1                              .value_counts()[train_trans.C1.value_counts() <= 400 ]                              .index), 'C1'] = "Others"

fig, ax = plt.subplots(2, 1, figsize=(24,12))

sb.countplot(x="C1", ax=ax[0], data=train_trans.loc[train_trans['isFraud'] == 0])
ax[0].set_title('C1 is not Fraud', fontsize=14)
sb.countplot(x="C1", ax=ax[1], data=train_trans.loc[train_trans['isFraud'] == 1])
ax[1].set_title('C1 is Fraud', fontsize=14)


# In[78]:


train_trans.loc[train_trans.C2.isin(train_trans.C2                              .value_counts()[train_trans.C2.value_counts() <= 350 ]                              .index), 'C2'] = "Others"

fig, ax = plt.subplots(2, 1, figsize=(24,12))

sb.countplot(x="C2", ax=ax[0], data=train_trans.loc[train_trans['isFraud'] == 0])
ax[0].set_title('C2 is not Fraud', fontsize=14)
sb.countplot(x="C2", ax=ax[1], data=train_trans.loc[train_trans['isFraud'] == 1])
ax[1].set_title('C2 is Fraud', fontsize=14)


# In[79]:


train_trans.loc[train_trans.C4.isin(train_trans.C4                              .value_counts()[train_trans.C4.value_counts() <= 400 ]                              .index), 'C4'] = "Others"

fig, ax = plt.subplots(2, 1, figsize=(20,8))

sb.countplot(x="C4", ax=ax[0], data=train_trans.loc[train_trans['isFraud'] == 0])
ax[0].set_title('C4 is not Fraud', fontsize=14)
sb.countplot(x="C4", ax=ax[1], data=train_trans.loc[train_trans['isFraud'] == 1])
ax[1].set_title('C4 is Fraud', fontsize=14)


# In[80]:


train_trans.loc[train_trans.D1.isin(train_trans.D1.value_counts()[train_trans.D1.value_counts() <= 2000 ].index), 'D1'] = "Others"
fig, ax = plt.subplots(2, 1, figsize=(20,12))

sb.countplot(x="D1", ax=ax[0], data=train_trans.loc[train_trans['isFraud'] == 0])
ax[0].set_title('D1 is not Fraud', fontsize=14)
sb.countplot(x="D1", ax=ax[1], data=train_trans.loc[train_trans['isFraud'] == 1])
ax[1].set_title('D1 is Fraud', fontsize=14)


# In[81]:


train_trans["M6"] = train_trans["M6"].fillna("Miss")
fig, ax = plt.subplots(2, 1, figsize=(20,12))

sb.countplot(x="M6", ax=ax[0], data=train_trans.loc[train_trans['isFraud'] == 0])
ax[0].set_title('M6 is not Fraud', fontsize=14)
sb.countplot(x="M6", ax=ax[1], data=train_trans.loc[train_trans['isFraud'] == 1])
ax[1].set_title('M6 is Fraud', fontsize=14)


# In[82]:


#Merge Transaction and Identity table
train_trans2 = pd.read_csv('creditcard_dataset2.csv/train_transaction.csv')
train = train_trans2.merge(train_id, how='left', on='TransactionID') 


# In[83]:


#Check for missing values
train.isnull().sum()


# In[84]:


#Fraud Counts
fraud_count = train["isFraud"].value_counts()
plt.figure()
plt.title("Fraud Class Counts")
plt.bar(fraud_count.index,fraud_count.values,color='purple')
plt.xticks([0,1], labels=["not fraud","fraud"])
plt.show()


# In[85]:


del train_id
del train_trans
del train_trans2


# ## Preprocess Data

# In[86]:


#Remove high missing value columns
columnsToDelete = []
for col in train.columns:
    if(train[col].isnull().sum()/len(train[col]) >= 0.4):
        print(col, "% NaN:", train[col].isnull().sum()/len(train[col]))
        columnsToDelete.append(col)


# In[87]:


train = train.drop(columns=columnsToDelete)


# In[88]:


cat_features = ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",                "P_emaildomain", "M6"]
num_features = []
for col in train.columns:
    if col not in cat_features:
        num_features.append(col)
num_features.remove('isFraud')
num_features.remove('TransactionID')
cat_features.remove("card1")

largecol = ["card1"]
features = num_features + cat_features + largecol
print(len(cat_features))
print(len(num_features))


# In[89]:


num_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='constant')),
        ('scaler', MinMaxScaler())  
    ]
)

cat_transformer = Pipeline(
    steps = [
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

large_transformer = Pipeline(
    steps = [
        ('scaler', MinMaxScaler())  
    ]
)

preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features),
        ('large', large_transformer, largecol)
    ]
)


# In[90]:


for col in largecol:
    train[col] = train[col].fillna(train[col].mode()[0])
    label = LabelEncoder()
    train[col] = label.fit_transform(train[col])
for col in cat_features:
    train[col] = train[col].fillna(train[col].mode()[0])


# In[91]:


from sklearn.model_selection import train_test_split
y_train = train.isFraud.values
x_train, x_test, y_train, y_test = train_test_split(train, y_train, stratify = y_train, test_size = 0.25, random_state = 5)


# In[92]:


preprocessor.fit(x_train[features])
x_train = preprocessor.transform(x_train[features])
x_test = preprocessor.transform(x_test[features])

print('X_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)


# In[93]:


#Dimension reduction
from sklearn.decomposition import TruncatedSVD
pca = TruncatedSVD(n_components = 200)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
var_explained = pca.explained_variance_ratio_.sum()
print(var_explained)


# In[94]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
def print_classification_result(true, predict):
    print(f"Accuracy Score: {accuracy_score(true, predict) * 100:.2f}%")
    print(f"Confusion Matrix: \n {confusion_matrix(true, predict)}\n")
    print(f"ROC_AUC_Score:{roc_auc_score(true, predict)}")


# ## Random Forest Classifier

# In[95]:


from sklearn.ensemble import RandomForestClassifier
randfclassifier = RandomForestClassifier(n_estimators = 100, criterion = "entropy")
randfclassifier.fit(x_train, y_train)
y_test_pred_randf = randfclassifier.predict(x_test)
print_classification_result(y_test, y_test_pred_randf)


# ## K Nearest Neighbor Classifier

# In[96]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = "minkowski")
knn.fit(x_train, y_train)
y_test_pred_KNN = knn.predict(x_test)
print_classification_result(y_test, y_test_pred_KNN)


# ## Naive Bayes Classifier

# In[97]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
y_test_pred_NB = nb.predict(x_test)
print_classification_result(y_test, y_test_pred_NB)


# ## Decision Tree Classifier

# In[98]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = "entropy")
tree.fit(x_train, y_train)
y_test_pred_tree = tree.predict(x_test)
print_classification_result(y_test, y_test_pred_tree)


# In[ ]:




