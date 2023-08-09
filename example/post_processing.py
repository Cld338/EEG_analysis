from sklearn.preprocessing import StandardScaler, RobustScaler
from private_tool import *
import pandas as pd
import numpy as np

raw_data_by_experiments = [pd.DataFrame(np.load(f"{currDir}/data/raw/dataframe_raw_{i}.npy")) for i in range(9)]

DF = raw_data_by_experiments[0]
for i in raw_data_by_experiments[1:]:
    DF = pd.concat([DF, i])

DF.columns = ["axis1", "axis2", "axis3", "label"]
scaler = StandardScaler()
scaler.fit(DF.loc[:, :"axis3"], DF.loc[:, "label"])
scaled_DF = pd.DataFrame(scaler.transform(DF.loc[:, :"axis3"]))
label_DF = pd.DataFrame(DF.loc[:, "label"], columns=["label"])
print(label_DF)
DF_Scaled = pd.concat([scaled_DF, label_DF.reset_index()], axis=1)

DF_Scaled.columns = [f"axis{i+1}" for i in range(4)] + ["label"]
LDA_DF_Scaled = pd.concat([DF_Scaled.loc[:, :"axis3"], DF_Scaled.loc[:, "label"]], axis=1)


import sklearn.svm as svm
import sklearn.metrics as mt
from sklearn.model_selection import cross_val_score, cross_validate

# SVM, kernel = 'linear'로 선형분리 진행
 
svm_clf =svm.SVC(kernel = 'linear', random_state=100)

# 교차검증

scores = cross_val_score(svm_clf, LDA_DF_Scaled.loc[:, :"axis3"], label_DF, cv = 5)
scores

pd.DataFrame(cross_validate(svm_clf, LDA_DF_Scaled.loc[:, :"axis3"], label_DF, cv =5))

print('교차검증 평균: ', scores.mean())

# SVM, kernel = 'rbf'로 비선형분리 진행
 
svm_clf =svm.SVC(kernel = 'rbf')

# 교차검증

scores = cross_val_score(svm_clf, LDA_DF_Scaled.loc[:, :"axis3"], label_DF, cv = 5)
scores

pd.DataFrame(cross_validate(svm_clf, LDA_DF_Scaled.loc[:, :"axis3"], label_DF, cv =5))

print('교차검증 평균: ', scores.mean())
print(DF_Scaled)