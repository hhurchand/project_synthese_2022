# -*- coding: utf-8 -*-
"""Methode Classique_12_05_2022.ipynb

"""

import numpy as np
import pandas as pd

# Importer les données
import glob
import matplotlib.image as image
import random
import collections
import matplotlib.pyplot as plt
#
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import pickle

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def load_data(raw=False):
    df = pd.DataFrame()
    if raw == True:
        path = "data/external/train.csv/ok_front"
        file_list_test_ok = glob.glob(path + "/*.*")
        path = "data/external/train.csv/def_front"
        file_list_test_not_ok = glob.glob(path + "/*.*")
        print(len(file_list_test_ok),len(file_list_test_not_ok))
        combined_file_list = []
        for i in file_list_test_ok:
            combined_file_list.append((i, 1))
        for j in file_list_test_not_ok:
            combined_file_list.append((j, 0))

        random.shuffle(combined_file_list)

        ind = 0

        for imag, target in combined_file_list:
            img = image.imread(imag)
            pixels_line_i = []
            for i in range(512):
                for j in range(512):
                    pixels_line_i.append(img[i][j][0])

            counter = dict(collections.Counter(pixels_line_i))
            counter["Target"] = target
            ind += 1
            df = pd.concat([df, pd.DataFrame(counter, index=[ind])], ignore_index=True)

        df.fillna(0, inplace=True)
        df.to_csv("dataframe_test.csv", index=False)

    elif raw == False:

        path = "src/models"
        df = pd.read_csv(path + "/dataframe_test.csv", header=0)

    return df


df = load_data(raw=False)
print(df.columns)
print(df.shape)
Y = df["Target"]
features = set(df.columns) - {"Target"}
X = df[features]

print(Y.shape)

print(X.shape)

# Examine the mean distribution of pixels by category
df_fault = df[df["Target"] == 0]
df_no_fault = df[df["Target"] == 1]

print("length fault",df_fault.shape[0])
print("length no fault",df_no_fault.shape[0])
df_fault_describe = df_fault.describe()
df_no_fault_describe = df_no_fault.describe()

# df_fault_describe.loc["mean",features]


# plt.plot(df_fault_describe.loc["mean",features])
# plt.plot(df_no_fault_describe.loc["mean",features])


# plt.plot(df_fault_describe.loc["std",features])
# plt.plot(df_no_fault_describe.loc["std",features])

# df.describe()


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

print(X_train.shape, X_test.shape)

print(Y_train.shape, Y_test.shape)

rs = RobustScaler().fit(X_train)

X_train_std = rs.transform(X_train)

X_test_std = rs.transform(X_test)



models = [SVC(C=100), RandomForestClassifier(),
          DecisionTreeClassifier(), AdaBoostClassifier(), KNeighborsClassifier(), GaussianNB()]

result_frame = {model.__class__.__name__: list() for model in models}
# result_frame


for model in models:
    model.fit(X_train_std, Y_train)
    y_predict = model.predict(X_test_std)
    accuracy_test = accuracy_score(Y_test, y_predict)
    result_frame[model.__class__.__name__].append(accuracy_test)
#    result_frame[model.__class__.__name__].append(
#        classification_report(Y_test, y_predict, output_dict=True)["weighted avg"]["f1-score"])

print(result_frame)

model_rf = RandomForestClassifier()
model_rf.fit(X_train_std, Y_train)

print(accuracy_score(model_rf.predict(X_test_std), Y_test))

# model_rf.feature_importances_.shape

forest_importances = pd.Series(model_rf.feature_importances_, index=features)

importances = model_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in model_rf.estimators_], axis=0)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
plt.figure(figsize=(20, 21))
fig.tight_layout()

# save serialized model and scaler
pickle.dump(model_rf, open("models/model_rf.pkl", 'wb'))
pickle.dump(rs, open("models/scaler.pkl", 'wb'))

print(confusion_matrix(Y_test, y_predict))

print(classification_report(Y_test, y_predict))

print(classification_report(Y_test, y_predict, output_dict=True)["weighted avg"]["f1-score"])


mlflow.set_experiment(experiment_name="experiment0")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
with mlflow.start_run():
#for model_name,accuracy in result_frame.items():
    mlflow.log_metric("Accuracy",result_frame[RandomForestClassifier().__class__.__name__][0])
    mlflow.sklearn.log_model(model_rf, "model")
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme