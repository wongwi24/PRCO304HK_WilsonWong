#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate


# ## Import and Analysis of data

# In[2]:


card_data = pd.read_csv('creditcard.csv\creditcard.csv')
print(card_data.shape)
X = card_data.iloc[:, :-1]
Y = card_data.iloc[:, -1]


# In[3]:


#Check missing values
weight_nf = card_data['Class'].value_counts()[0] / len(card_data)
weight_f = card_data['Class'].value_counts()[1] / len(card_data)
nf = card_data["Class"].value_counts()[0]
f = card_data["Class"].value_counts()[1]
print(f"Non-Fraud weight: {weight_nf}")
print(f"Fraud weight: {weight_f}")
print(f"Non Fraud Count: {nf}")
print(f"Fraud Count: {f}")


# In[4]:


card_data.describe()


# In[5]:


# Check for null values
card_data.isnull().sum()


# In[6]:


#Plot Fraud vs Not Fraud transaction counts
fraud_count = card_data["Class"].value_counts()
plt.figure()
plt.title("Fraud Class Counts")
plt.bar(fraud_count.index,fraud_count.values,color='purple')
plt.xticks([0,1], labels=["not fraud","fraud"])
plt.show()


# In[7]:


#Plot features correlation
plt.figure(figsize = (13,13))
corr = card_data.corr()
plt.title("Features Correlation")
sb.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.show


# In[8]:


#Transaction Time density plot
notFraud_time = card_data.loc[card_data["Class"]==0]
fraud_time = card_data.loc[card_data["Class"]==1]
plt.figure()
sb.kdeplot(data = notFraud_time["Time"], label = "Class 0")
sb.kdeplot(data = fraud_time["Time"], label = "Class 1")
plt.legend()
plt.show()


# In[9]:


sb.boxplot(x = "Class", y = "Amount", hue = "Class", data = card_data, showfliers = False)
plt.show()


# In[10]:


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

# In[11]:


#Split dataset into test train and valid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify = Y, test_size = 0.25, random_state = 5)


# In[12]:


ct = ColumnTransformer([
        ('std', StandardScaler(), ['Amount', 'Time'])
    ], remainder='passthrough')
x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)


# In[13]:


weight_nf = y_train.value_counts()[0] / len(y_train)
weight_f = y_train.value_counts()[1] / len(y_train)
print(f"Non-Fraud weight: {weight_nf}")
print(f"Fraud weight: {weight_f}")


# In[14]:


print(f"Train Data shape: {x_train.shape} Train Class Data shape: {y_train.shape}")
print(f"Test Data shape: {x_test.shape} Test Class Data shape: {y_test.shape}")


# ## Random Forest Classifier

# In[16]:


from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_score, matthews_corrcoef


def print_classification_result(true, predict):
    print(f"Accuracy Score\n: {accuracy_score(true, predict) * 100:.2f}%")
    print(f"Confusion Matrix: \n {confusion_matrix(true, predict)}\n")
    print(f"MCC_Score\n:{matthews_corrcoef(true, predict)}\n")
    print(f"f1_score: \n {f1_score(true, predict)}\n")
    print(f"recall_score: \n {recall_score(true, predict)}\n")
    print(f"Precision_Score\n:{precision_score(true, predict)}")

def print_cross_val_result(cv):
    print(f"Accuracy Score: \n{cv['test_accuracy'].mean()}\n")
    print(f"f1_score: \n{cv['test_f1'].mean()}\n")
    print(f"recall_score: \n{cv['test_recall'].mean()}\n")
    print(f"Precision_Score: \n{cv['test_precision'].mean()}")


# In[45]:


from sklearn.ensemble import RandomForestClassifier
randfclassifier = RandomForestClassifier(n_estimators = 100, criterion = "gini")
randfclassifier.fit(x_train, y_train)
y_test_pred_randf = randfclassifier.predict(x_test)
print_classification_result(y_test, y_test_pred_randf)


# In[46]:


plt.figure(figsize = (7,5))
cm = sb.heatmap(confusion_matrix(y_test, y_test_pred_randf), annot=True, cmap='Blues', fmt='d')
cm.set_title('Random Forest')
plt.show()


# In[18]:


randfclassifier = RandomForestClassifier(n_estimators = 100, criterion = "gini")
cv = cross_validate(randfclassifier, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[19]:


print_cross_val_result(cv)


# In[20]:


randfclassifier = RandomForestClassifier(n_estimators = 200, criterion = "gini")
cv = cross_validate(randfclassifier, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[21]:


print_cross_val_result(cv)


# In[22]:


randfclassifier = RandomForestClassifier(n_estimators = 300, criterion = "gini")
cv = cross_validate(randfclassifier, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[23]:


print_cross_val_result(cv)


# In[24]:


randfclassifier = RandomForestClassifier(n_estimators = 100, criterion = "gini", bootstrap = False)
cv = cross_validate(randfclassifier, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[25]:


print_cross_val_result(cv)


# ## Kernel Support Vector Machine

# In[19]:


from sklearn.svm import SVC
KSVM = SVC(kernel = "poly", C = 1)
KSVM.fit(x_train, y_train)
y_test_pred_KSVM = KSVM.predict(x_test)
print_classification_result(y_test, y_test_pred_KSVM)


# In[20]:


plt.figure(figsize = (7,5))
cm = sb.heatmap(confusion_matrix(y_test, y_test_pred_KSVM), annot=True, cmap="Blues", fmt="g")
cm.set_title('KSVM')
plt.show()


# In[22]:


KSVM = SVC(kernel = "rbf", C = 1)
cv = cross_validate(KSVM, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[23]:


print_cross_val_result(cv)


# In[24]:


KSVM = SVC(kernel = "rbf", C = 10)
cv = cross_validate(KSVM, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[25]:


print_cross_val_result(cv)


# In[15]:


from sklearn.svm import SVC
KSVM = SVC(kernel = "rbf", C = 0.1)
cv = cross_validate(KSVM, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[18]:


print_cross_val_result(cv)


# In[26]:


KSVM = SVC(kernel = "poly")
cv = cross_validate(KSVM, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[27]:


print_cross_val_result(cv)


# In[28]:


KSVM = SVC(kernel = "linear")
cv = cross_validate(KSVM, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[29]:


print_cross_val_result(cv)


# ## K Nearest Neighbors

# In[30]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = "minkowski")
knn.fit(x_train, y_train)
y_test_pred_KNN = knn.predict(x_test)
print_classification_result(y_test, y_test_pred_KNN)


# In[31]:


plt.figure(figsize = (7,5))
cm = sb.heatmap(confusion_matrix(y_test, y_test_pred_KNN), annot=True, cmap='Blues', fmt='d')
cm.set_title('K Nearest Neighbors')
plt.show()


# In[32]:


knn = KNeighborsClassifier(n_neighbors = 4, p = 2, metric = "minkowski")
cv = cross_validate(knn, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[33]:


print_cross_val_result(cv)


# In[34]:


knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = "minkowski")
cv = cross_validate(knn, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[35]:


print_cross_val_result(cv)


# In[36]:


knn = KNeighborsClassifier(n_neighbors = 6, p = 2, metric = "minkowski")
cv = cross_validate(knn, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[37]:


print_cross_val_result(cv)


# In[43]:


knn = KNeighborsClassifier(n_neighbors = 5, p = 1, metric = "minkowski")
cv = cross_validate(knn, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[44]:


print_cross_val_result(cv)


# ## Naive Bayes Classifier

# In[38]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
y_test_pred_NB = nb.predict(x_test)
print_classification_result(y_test, y_test_pred_NB)


# In[39]:


plt.figure(figsize = (7,5))
cm = sb.heatmap(confusion_matrix(y_test, y_test_pred_NB), annot=True, cmap='Blues', fmt='d')
cm.set_title('Naive Bayes')
plt.show()


# ## Decision Tree Classifier

# In[53]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 10)
tree.fit(x_train, y_train)
y_test_pred_tree = tree.predict(x_test)
print(f"max depth of tree: {tree.tree_.max_depth}\n")
print_classification_result(y_test, y_test_pred_tree)


# In[54]:


plt.figure(figsize = (7,5))
cm = sb.heatmap(confusion_matrix(y_test, y_test_pred_tree), annot=True, cmap='Blues', fmt='d')
cm.set_title('Decision Tree')
plt.show()


# In[42]:


tree = DecisionTreeClassifier(criterion = "entropy")
cv = cross_validate(tree, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[43]:


print_cross_val_result(cv)


# In[44]:


tree = DecisionTreeClassifier(criterion = "gini")
cv = cross_validate(tree, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[45]:


print_cross_val_result(cv)


# In[46]:


tree = DecisionTreeClassifier(criterion = "gini", splitter = "random")
cv = cross_validate(tree, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[47]:


print_cross_val_result(cv)


# In[48]:


tree = DecisionTreeClassifier(criterion = "gini", max_features = "sqrt")
cv = cross_validate(tree, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[49]:


print_cross_val_result(cv)


# In[57]:


tree = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 15)
cv = cross_validate(tree, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[58]:


print_cross_val_result(cv)


# In[49]:


tree = DecisionTreeClassifier(criterion = "gini", min_samples_leaf = 10)
cv = cross_validate(tree, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[50]:


print_cross_val_result(cv)


# In[51]:


tree = DecisionTreeClassifier(criterion = "gini", max_depth = 20)
cv = cross_validate(tree, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[52]:


print_cross_val_result(cv)


# ## Logistics Regression

# In[21]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression(C=10, class_weight=None)
log.fit(x_train, y_train)
y_test_pred_log = log.predict(x_test)
print_classification_result(y_test, y_test_pred_log)


# In[22]:


plt.figure(figsize = (7,5))
cm = sb.heatmap(confusion_matrix(y_test, y_test_pred_log), annot=True, cmap='Blues', fmt='d')
cm.set_title('Logistic Regression')
plt.show()


# In[25]:


log = LogisticRegression(C=1, class_weight=None)
cv = cross_validate(log, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[26]:


print_cross_val_result(cv)


# In[27]:


log = LogisticRegression(C=0.1, class_weight=None)
cv = cross_validate(log, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[28]:


print_cross_val_result(cv)


# In[29]:


log = LogisticRegression(C=10, class_weight=None)
cv = cross_validate(log, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[30]:


print_cross_val_result(cv)


# In[31]:


log = LogisticRegression(C=1, class_weight="balanced")
cv = cross_validate(log, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[32]:


print_cross_val_result(cv)


# In[39]:


log = LogisticRegression(C=1, class_weight=None, solver="saga", penalty="l2")
cv = cross_validate(log, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[40]:


print_cross_val_result(cv)


# In[41]:


log = LogisticRegression(C=1, class_weight=None, solver="saga", penalty="l1")
cv = cross_validate(log, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[42]:


print_cross_val_result(cv)


# In[23]:


log = LogisticRegression(C=1, class_weight=None, solver="newton-cg", penalty="l2")
cv = cross_validate(log, x_train, y_train, cv=5, 
               scoring=('accuracy','f1','recall','precision'), 
               return_train_score=True)


# In[24]:


print_cross_val_result(cv)


# In[ ]:




