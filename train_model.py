#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
#import pickle
import joblib

# # Distribution des pixels

# In[3]:


df = pd.read_csv("src/models/dataframe_test.csv",header=0)

# In[4]:


Y = df["Target"]
features = set(df.columns) - {"Target"}
X = df[features]

# In[5]:


# Examine the mean distribution of pixels by category
df_fault = df[df["Target"] == 0]
df_no_fault = df[df["Target"] == 1]

df_fault_describe = df_fault.describe()
df_no_fault_describe = df_no_fault.describe()

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.31)

print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)



rs = RobustScaler().fit(X_train)



filename_scaler = 'models/scaler.sav'
joblib.dump(rs,filename_scaler)
rs1 = joblib.load("models/scaler.sav")

X_train_std = rs.transform(X_train)
X_train_std_x = rs1.transform(X_train)

# In[12]:
print("real",X_train_std.min(),X_train_std.max())
print("copied",X_train_std_x.min(),X_train_std_x.max())

X_test_std = rs.transform(X_test)
X_test_std_x = rs1.transform(X_test)
# In[13]:
print("real_test",X_test_std.min(),X_test_std.max())
print("copied_test",X_test_std_x.min(),X_test_std_x.max())

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# In[14]:


# models = [SVC(C=100),RandomForestClassifier(),LogisticRegression(max_iter=300),\
#          DecisionTreeClassifier(),AdaBoostClassifier(),KNeighborsClassifier(),GaussianNB()]
model_0 = RandomForestClassifier()

filename_model = 'models/model_rf.sav'
joblib.dump(model_0.fit(X_train_std, Y_train), filename_model)
models_1 = [joblib.load("models/model_rf.sav")]

# In[15]:


#result_frame = {model.__class__.__name__: list() for model in models}

# In[16]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# In[17]:


#for model in models:
#    model.fit(X_train_std, Y_train)
y_predict = model_0.predict(X_test_std)
accuracy_test = accuracy_score(Y_test, y_predict)
#result_frame[model.__class__.__name__].append(accuracy_test)
#result_frame[model.__class__.__name__].append(
#classification_report(Y_test, y_predict, output_dict=True)["weighted avg"]["f1-score"])

# In[18]:

#model_rf = models[0]
#model_rf = RandomForestClassifier()
#model_rf.fit(X_train_std, Y_train)
y_predict_1 = models_1[0].predict(X_test_std_x)
print("accuracy_copied",accuracy_score(Y_test, y_predict_1))
# In[19]:


print("accuracy True",accuracy_score(model_0.predict(X_test_std), Y_test))

# forest_importances = pd.Series(model_rf.feature_importances_, index=features)
#
# importances = model_rf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in model_rf.estimators_], axis=0)
#
# fig, ax = plt.subplots()
# forest_importances.plot.bar(ax=ax)
# ax.set_title("Feature importances")
# ax.set_ylabel("Mean decrease in impurity")
# plt.figure(figsize=(20,20))
# fig.tight_layout()

# In[20]:

# open a file, where you ant to store the model
#pickle.dump(model_rf, open("models/model_rf.pkl", 'wb'))
#pickle.dump(rs, open("models/scaler.pkl", 'wb'))
# with open('model_rf.pickle', 'wb') as f:
#     pickle.dump(model_rf, f)
#
# with open('scaler.pickle', 'wb') as g:
#     pickle.dump(rs, g)
##filename_model = 'models/model_rf.sav'
##joblib.dump(model_rf, filename_model)

##filename_scaler = 'models/scaler.sav'
##joblib.dump(rs, filename_scaler)

print(accuracy_test)

# In[21]:


print(confusion_matrix(Y_test, y_predict))

# In[22]:


print(classification_report(Y_test, y_predict))

# In[23]:


print("classification report", classification_report(Y_test, y_predict, output_dict=True)["weighted avg"]["f1-score"])







