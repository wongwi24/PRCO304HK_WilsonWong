#!/usr/bin/env python
# coding: utf-8

# In[155]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
train_id = pd.read_csv('creditcard_dataset2.csv/train_identity.csv')
train_trans = pd.read_csv('creditcard_dataset2.csv/train_transaction.csv')


# In[156]:


print(train_id.shape)
print(train_trans.shape)


# In[157]:


#Merge Transaction and Identity table
train = train_trans.merge(train_id, how='left', on='TransactionID') 
#Check for missing values
train.isnull().sum()


# In[158]:


del train_id
del train_trans


# In[159]:


#Remove high missing value columns
columnsToDelete = []
for col in train.columns:
    if(train[col].isnull().sum()/len(train[col]) >= 0.8):
        print(col, "% NaN:", train[col].isnull().sum()/len(train[col]))
        columnsToDelete.append(col)


# In[160]:


train = train.drop(columns=columnsToDelete)


# In[161]:


v_columns = []
for col in ['V'+str(x) for x in range(1,340)]:
    if col in train.columns:
        v_columns.append(col)


# In[162]:


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


# In[163]:


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
us_emails = ['gmail', 'net', 'edu']

for c in ['P_emaildomain', 'R_emaildomain']:
    train[c] = train[c].map(emails)


# In[164]:


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


# In[165]:


def make_hour_feature(df, tname='TransactionDT'):
    hours = df[tname] / (3600)        
    encoded_hours = np.floor(hours) % 24
    return encoded_hours

train['hours'] = make_hour_feature(train)
num_features.remove('TransactionDT')
cat_onehot_features.append('hours')


# In[166]:


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


# In[167]:


for col in cat_label_features:
    train[col] = train[col].fillna(train[col].mode()[0])

le = LabelEncoder()
train[cat_label_features] = train[["card1","card2","card3","card5", "addr1", "addr2", "id_13","id_17",
                                   "id_19","id_20","DeviceInfo"]].apply(le.fit_transform)


# In[168]:


from sklearn.model_selection import train_test_split
y = train.isFraud.values
x_train, x_test, y_train, y_test = train_test_split(train, y, stratify = y, test_size = 0.25, random_state = 5)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, stratify = y_train, test_size = 0.2, random_state = 5)


# In[169]:


x_train_cat_label = x_train[cat_label_features]
x_test_cat_label = x_test[cat_label_features]
x_valid_cat_label = x_valid[cat_label_features]
mms = MinMaxScaler()
x_train_cat_label = mms.fit_transform(x_train_cat_label)
x_test_cat_label = mms.transform(x_test_cat_label)
x_valid_cat_label = mms.transform(x_valid_cat_label)
print(x_train_cat_label.shape)


# In[170]:


preprocessor_c.fit(x_train[cat_onehot_features])
x_train_cat_onehot = preprocessor_c.transform(x_train[cat_onehot_features])
x_test_cat_onehot = preprocessor_c.transform(x_test[cat_onehot_features])
x_valid_cat_onehot = preprocessor_c.transform(x_valid[cat_onehot_features])


# In[171]:


x_train_cat_onehot = x_train_cat_onehot.toarray()
x_test_cat_onehot = x_test_cat_onehot.toarray()
x_valid_cat_onehot = x_valid_cat_onehot.toarray()
print(x_train_cat_onehot.shape)


# In[172]:


preprocessor_v.fit(x_train[v_columns])
x_train_v = preprocessor_v.transform(x_train[v_columns])
x_test_v = preprocessor_v.transform(x_test[v_columns])
x_valid_v = preprocessor_v.transform(x_valid[v_columns])
print(x_train_v.shape)


# In[173]:


#Dimension reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)
x_train_v = pca.fit_transform(x_train_v)
x_test_v = pca.transform(x_test_v)
x_valid_v = pca.transform(x_valid_v)
var_explained = pca.explained_variance_ratio_.sum()
print(x_train_v.shape)


# In[174]:


preprocessor_num.fit(x_train[num_features])
x_train_num = preprocessor_num.transform(x_train[num_features])
x_test_num = preprocessor_num.transform(x_test[num_features])
x_valid_num = preprocessor_num.transform(x_valid[num_features])
print(x_train_num.shape)
print(x_test_num.shape)
print(x_valid_num.shape)


# In[175]:


x_train_num = x_train_num.astype('float32')
x_test_num = x_test_num.astype('float32')
x_valid_num = x_valid_num.astype('float32')
x_train = np.concatenate((x_train_num, x_train_cat_label, x_train_v, x_train_cat_onehot), axis = 1)
x_test = np.concatenate((x_test_num, x_test_cat_label, x_test_v, x_test_cat_onehot), axis = 1)
x_valid = np.concatenate((x_valid_num, x_valid_cat_label, x_valid_v, x_valid_cat_onehot), axis = 1)
print(x_train.shape)
print(x_test.shape)
print(x_valid.shape)


# In[176]:


from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_score, matthews_corrcoef


def print_classification_result(true, predict):
    print(f"Accuracy Score: {accuracy_score(true, predict) * 100:.2f}%")
    print(f"Confusion Matrix: \n {confusion_matrix(true, predict)}\n")
    print(f"MCC_Score:{matthews_corrcoef(true, predict)}\n")
    print(f"f1_score: \n {f1_score(true, predict)}\n")
    print(f"recall_score: \n {recall_score(true, predict)}\n")
    print(f"Precision_Score:{precision_score(true, predict)}")


# ## Convolution Neural Network

# In[177]:


x_train = np.array(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = np.array(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)
x_valid = np.array(x_valid).reshape(x_valid.shape[0], x_valid.shape[1], 1)


# In[178]:


cnn = tf.keras.models.Sequential()


# In[179]:


cnn.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 3, activation = 'relu', input_shape = [227, 1]))
cnn.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 1))


# In[180]:


cnn.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool1D(pool_size = 2, strides = 1))


# In[181]:


cnn.add(tf.keras.layers.Flatten())


# In[182]:


cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))


# In[183]:


cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# In[184]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[185]:


loss = cnn.fit(x_train, y_train, batch_size = 32, epochs = 8, verbose = 1, validation_data = (x_valid, y_valid))


# In[186]:


plt.plot(loss.history['loss'])
plt.plot(loss.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[187]:


y_pred = cnn.predict(x_test)
y_pred = np.round(y_pred)
print_classification_result(y_test, y_pred)


# In[ ]:




