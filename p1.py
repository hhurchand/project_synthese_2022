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
# from sklearn.metrics import ConfusionMatrixDisplay


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import os
import pickle
import dataframe_image as dfi

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

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



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

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

df_result = pd.DataFrame(result_frame,index=["Accuracy"])
dfi.export(df_result,"mytable.png")

model_rf = RandomForestClassifier()
model_rf.fit(X_train_std, Y_train)

y_pred = model_rf.predict(X_test_std)
accuracy_of_model = accuracy_score(y_pred, Y_test)
print(accuracy_of_model)

score_f1 = classification_report(Y_test, y_pred, output_dict=True)["weighted avg"]["f1-score"]
cm = confusion_matrix(Y_test, y_pred)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#disp.plot()
#plt.savefig('confusion_matrix.png')
os.environ['MLFLOW_TRACKING_USERNAME'] = "h.hurchand"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "c849831fd1e33c252105db9c11369695ee50a48a"
mlflow.set_tracking_uri("https://dagshub.com/h.hurchand/dagshub_integration.mlflow")
mlflow.log_metric("accuracy SVM",df_result.iloc[0,0])
mlflow.log_metric("accuracy RF",df_result.iloc[0,1])
mlflow.log_metric("accuracy DT",df_result.iloc[0,2])
mlflow.log_metric("accuracy AdaB",df_result.iloc[0,3])
mlflow.log_metric("accuracy kNN",df_result.iloc[0,4])
mlflow.log_metric("accuracy NB",df_result.iloc[0,5])
#mlflow.log_artifact("confusion_matrix.png")
mlflow.log_artifact("mytable.png")











