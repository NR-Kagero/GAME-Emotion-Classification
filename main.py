import os
import numpy as np
import xgboost as xgb
from Preprocessing import *
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, balanced_accuracy_score

from Preprocessing import *

path = "C:\\Users\\Kagero\\PycharmProjects\\GAME-Emotion-Classification\\GAMEEMO"

x, y = data_preparing_preprocessing_reshaping_with_overlap(path, 8, 7)
print(x.shape, y.shape)
XGBoost = xgb.XGBClassifier(n_estimators=500, learning_rate=0.2, max_depth=15, reg_alpha=0.5, reg_lambda=0.5)
metrics = [accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score]
results = k_fold_test(K=7, X=x, Y=y, metrics=metrics, model=XGBoost)
