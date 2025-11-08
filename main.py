import os
import numpy as np
import xgboost as xgb
from Preprocessing import *
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, balanced_accuracy_score

from Preprocessing import *

# path = "C:\\Users\\Kagero\\PycharmProjects\\GAME-Emotion-Classification\\GAMEEMO"
#
# x, y = data_preparing_preprocessing_reshaping_with_overlap(path, 8, 7)
# print(x.shape, y.shape)
# XGBoost = xgb.XGBClassifier(n_estimators=500, learning_rate=0.2, max_depth=10, reg_alpha=0.1, reg_lambda=0.5)
# metrics = [accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score]
# results = k_fold_test(K=7, X=x, Y=y, metrics=metrics, model=XGBoost)
# for i in results:
#     print(i)
#     print(results[i])
results = {1: {'Metric accuracy_score result ': 0.8769463667820069,
               'Metric balanced_accuracy_score result ': 0.8784757505179996,
               'Metric f1_score result ': 0.8824439576964764, 'Metric recall_score result ': 0.8784757505179996,
               'Metric precision_score result ': 0.9116402363534912},
           2: {'Metric accuracy_score result ': 0.8769463667820069,
               'Metric balanced_accuracy_score result ': 0.875983515506874,
               'Metric f1_score result ': 0.8833592585102107, 'Metric recall_score result ': 0.875983515506874,
               'Metric precision_score result ': 0.9160304279271008},
           3: {'Metric accuracy_score result ': 0.8752162629757786,
               'Metric balanced_accuracy_score result ': 0.8740834140368053,
               'Metric f1_score result ': 0.8816258984925424, 'Metric recall_score result ': 0.8740834140368053,
               'Metric precision_score result ': 0.9134748063049233},
           4: {'Metric accuracy_score result ': 0.8685121107266436,
               'Metric balanced_accuracy_score result ': 0.8693035321970856,
               'Metric f1_score result ': 0.8748489439446425, 'Metric recall_score result ': 0.8693035321970856,
               'Metric precision_score result ': 0.90858541870837},
           5: {'Metric accuracy_score result ': 0.8799740484429066,
               'Metric balanced_accuracy_score result ': 0.879912000551943, 'Metric f1_score result ': 0.88571731874731,
               'Metric recall_score result ': 0.879912000551943, 'Metric precision_score result ': 0.9154495946159488},
           6: {'Metric accuracy_score result ': 0.8650519031141869,
               'Metric balanced_accuracy_score result ': 0.8664162071714636,
               'Metric f1_score result ': 0.8711523127857014, 'Metric recall_score result ': 0.8664162071714636,
               'Metric precision_score result ': 0.9053065846126636},
           7: {'Metric accuracy_score result ': 0.8760813148788927,
               'Metric balanced_accuracy_score result ': 0.8747837861709604,
               'Metric f1_score result ': 0.8821183693498913, 'Metric recall_score result ': 0.8747837861709604,
               'Metric precision_score result ': 0.9143479623144102}}
k_fold_results_plots(results=results)
