#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE


# In[2]:


#Merge Transaction and Identity table
train_id = pd.read_csv('creditcard_dataset2.csv/train_identity.csv')
train_trans = pd.read_csv('creditcard_dataset2.csv/train_transaction.csv')
train = train_trans.merge(train_id, how='left', on='TransactionID')


# In[3]:


print(train_id.shape)
print(train_trans.shape)


# In[4]:


#Check for missing values
weight_nf = train['isFraud'].value_counts()[0] / len(train)
weight_f = train['isFraud'].value_counts()[1] / len(train)
print(f"Non-Fraud weight: {weight_nf}")
print(f"Fraud weight: {weight_f}")


# In[5]:


train.isnull().sum()


# In[6]:


#Fraud Counts
fraud_count = train["isFraud"].value_counts()
plt.figure()
plt.title("Fraud Class Counts")
plt.bar(fraud_count.index,fraud_count.values,color='purple')
plt.xticks([0,1], labels=["not fraud","fraud"])
plt.show()


# In[7]:


del train_id
del train_trans


# ## Preprocess Data

# In[8]:


#Remove high missing value columns
columnsToDelete = []
for col in train.columns:
    if(train[col].isnull().sum()/len(train[col]) >= 0.8):
        print(col, "% NaN:", train[col].isnull().sum()/len(train[col]))
        columnsToDelete.append(col)


# In[9]:


train = train.drop(columns=columnsToDelete)


# In[10]:


v_columns = []
for col in ['V'+str(x) for x in range(1,340)]:
    if col in train.columns:
        v_columns.append(col)


# In[11]:


cat_label_features = ["card1","card2","card3","card5", "addr1", "addr2", "id_13","id_17","id_19","id_20","id_31","DeviceInfo"]
num_features = []
cat_onehot_features = ["ProductCD","card4","card6", "M1","M2","M3","M4","M5","M6","M7","M8","M9","id_12","id_15",
                      "id_16","id_28","id_29","id_35","id_36","id_37","id_38","DeviceType","P_emaildomain",
                       "R_emaildomain"]
for col in train.columns:
    if col not in cat_label_features and col not in v_columns and col not in cat_onehot_features:
        num_features.append(col)
num_features.remove('isFraud')
num_features.remove('TransactionID')

print(len(cat_onehot_features))
print(len(cat_label_features))
print(len(num_features))


# In[12]:


emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other',
          'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft',
          'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 
          'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other',
          'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo',
          'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo',
          'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo',
          'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',
          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',
          'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',
          'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other',
          'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

for c in ['P_emaildomain', 'R_emaildomain']:
    train[c] = train[c].map(emails)


# In[13]:


train["latest_browser"] = np.zeros(train.shape[0])

def setBrowser(df):
    df.loc[df["id_31"]=="samsung browser 7.0",'latest_browser']=1
    df.loc[df["id_31"]=="opera 53.0",'latest_browser']=1
    df.loc[df["id_31"]=="mobile safari 10.0",'latest_browser']=1
    df.loc[df["id_31"]=="google search application 49.0",'latest_browser']=1
    df.loc[df["id_31"]=="firefox 60.0",'latest_browser']=1
    df.loc[df["id_31"]=="edge 17.0",'latest_browser']=1
    df.loc[df["id_31"]=="chrome 69.0",'latest_browser']=1
    df.loc[df["id_31"]=="chrome 67.0 for android",'latest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for android",'latest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for ios",'latest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0",'latest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for android",'latest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for ios",'latest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0",'latest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for android",'latest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for ios",'latest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0",'latest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for android",'latest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for ios",'latest_browser']=1
    return df

train = setBrowser(train)
cat_label_features.remove('id_31')
cat_onehot_features.append('latest_browser')
train.drop(columns='id_31')


# In[14]:


def make_hour_feature(df, tname='TransactionDT'):
    hours = df[tname] / (3600)        
    encoded_hours = np.floor(hours) % 24
    return encoded_hours


train['hours'] = make_hour_feature(train)
num_features.remove('TransactionDT')
cat_onehot_features.append('hours')


# In[15]:


num_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())  
    ]
)

v_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='constant')),
        ('scaler', StandardScaler())
    ]
)

cat_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder())
    ]
)

preprocessor_c = ColumnTransformer(
    transformers = [
        ('cat', cat_transformer, cat_onehot_features)
    ]
)

preprocessor_v = ColumnTransformer(
    transformers = [
        ('v', v_transformer, v_columns)
    ]
)

preprocessor_num = ColumnTransformer(
    transformers = [
        ('num', num_transformer, num_features)
    ]
)


# In[16]:


for col in cat_label_features:
    train[col] = train[col].fillna(train[col].mode()[0])

le = LabelEncoder()
train[cat_label_features] = train[["card1","card2","card3","card5", "addr1", "addr2",
                                   "id_13","id_17","id_19","id_20","DeviceInfo"]].apply(le.fit_transform)


# In[17]:


from sklearn.model_selection import train_test_split
y = train.isFraud.values
x_train, x_test, y_train, y_test = train_test_split(train, y, test_size = 0.25, random_state = 5)


# In[18]:


x_train_cat_label = x_train[cat_label_features]
x_test_cat_label = x_test[cat_label_features]
mms = MinMaxScaler()
x_train_cat_label = mms.fit_transform(x_train_cat_label)
x_test_cat_label = mms.transform(x_test_cat_label)
print(x_train_cat_label.shape)


# In[19]:


preprocessor_c.fit(x_train[cat_onehot_features])
x_train_cat_onehot = preprocessor_c.transform(x_train[cat_onehot_features])
x_test_cat_onehot = preprocessor_c.transform(x_test[cat_onehot_features])


# In[20]:


x_train_cat_onehot = x_train_cat_onehot.toarray()
x_test_cat_onehot = x_test_cat_onehot.toarray()
print(x_train_cat_onehot.shape)


# In[21]:


preprocessor_v.fit(x_train[v_columns])
x_train_v = preprocessor_v.transform(x_train[v_columns])
x_test_v = preprocessor_v.transform(x_test[v_columns])
print(x_train_v.shape)


# In[22]:


#Dimension reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)
x_train_v = pca.fit_transform(x_train_v)
x_test_v = pca.transform(x_test_v)
var_explained = pca.explained_variance_ratio_.sum()
print(x_train_v.shape)


# In[23]:


preprocessor_num.fit(x_train[num_features])
x_train_num = preprocessor_num.transform(x_train[num_features])
x_test_num = preprocessor_num.transform(x_test[num_features])
print(x_train_num.shape)
print(x_test_num.shape)


# In[24]:


x_train_num = x_train_num.astype('float32')
x_test_num = x_test_num.astype('float32')
x_train = np.concatenate((x_train_num, x_train_cat_label, x_train_v, x_train_cat_onehot), axis = 1)
x_test = np.concatenate((x_test_num, x_test_cat_label, x_test_v, x_test_cat_onehot), axis = 1)
print(x_train.shape)
print(x_test.shape)


# In[25]:


smote = SMOTE(sampling_strategy='minority')
x_train, y_train = smote.fit_resample(x_train, y_train)
print(x_train.shape)


# In[26]:


from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_score, matthews_corrcoef


def print_classification_result(true, predict):
    print(f"Accuracy Score\n: {accuracy_score(true, predict) * 100:.2f}%")
    print(f"Confusion Matrix: \n {confusion_matrix(true, predict)}\n")
    print(f"MCC_Score\n:{matthews_corrcoef(true, predict)}\n")
    print(f"f1_score: \n {f1_score(true, predict)}\n")
    print(f"recall_score: \n {recall_score(true, predict)}\n")
    print(f"Precision_Score\n:{precision_score(true, predict)}")


# ## Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
randfclassifier = RandomForestClassifier(n_estimators = 100, criterion = "entropy")
randfclassifier.fit(x_train, y_train)
y_test_pred_randf = randfclassifier.predict(x_test)
print_classification_result(y_test, y_test_pred_randf)


# In[ ]:


from sklearn.metrics import plot_confusion_matrix
plt.figure(figsize = (7,5))
cm = sb.heatmap(confusion_matrix(y_test, y_test_pred_randf), annot=True, cmap='Blues', fmt='d')
cm.set_title('Random Forest')
plt.show()


# ## K Nearest Neighbor Classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = "minkowski")
knn.fit(x_train, y_train)
y_test_pred_KNN = knn.predict(x_test)
print_classification_result(y_test, y_test_pred_KNN)


# In[ ]:


from sklearn.metrics import plot_confusion_matrix
plt.figure(figsize = (7,5))
cm = sb.heatmap(confusion_matrix(y_test, y_test_pred_KNN), annot=True, cmap='Blues', fmt='d')
cm.set_title('K Nearest Neighbors')
plt.show()


# ## Naive Bayes Classifier

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
y_test_pred_NB = nb.predict(x_test)
print_classification_result(y_test, y_test_pred_NB)


# In[ ]:


from sklearn.metrics import plot_confusion_matrix
plt.figure(figsize = (7,5))
cm = sb.heatmap(confusion_matrix(y_test, y_test_pred_NB), annot=True, cmap='Blues', fmt='d')
cm.set_title('Naive Bayes')
plt.show()


# ## Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = "entropy")
tree.fit(x_train, y_train)
y_test_pred_tree = tree.predict(x_test)
print_classification_result(y_test, y_test_pred_tree)


# In[ ]:


from sklearn.metrics import plot_confusion_matrix
plt.figure(figsize = (7,5))
cm = sb.heatmap(confusion_matrix(y_test, y_test_pred_tree), annot=True, cmap='Blues', fmt='d')
cm.set_title('Decision Tree')
plt.show()

