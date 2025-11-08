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
results = {1: {'Metric accuracy_score result ': 0.9467429577464789,
               'Metric balanced_accuracy_score result ': 0.9466568636186341,
               'Metric f1_score result ': 0.9479058151176654, 'Metric recall_score result ': 0.9466568636186341,
               'Metric precision_score result ': 0.9546717727834749},
           2: {'Metric accuracy_score result ': 0.9445422535211268,
               'Metric balanced_accuracy_score result ': 0.944666784292768,
               'Metric f1_score result ': 0.9458622534291516, 'Metric recall_score result ': 0.944666784292768,
               'Metric precision_score result ': 0.9534290070069261},
           3: {'Metric accuracy_score result ': 0.9476232394366197,
               'Metric balanced_accuracy_score result ': 0.9480690085291266,
               'Metric f1_score result ': 0.9487258745376468, 'Metric recall_score result ': 0.9480690085291266,
               'Metric precision_score result ': 0.9551549072495709},
           4: {'Metric accuracy_score result ': 0.9496038732394366,
               'Metric balanced_accuracy_score result ': 0.9496742116800192,
               'Metric f1_score result ': 0.9509438113216173, 'Metric recall_score result ': 0.9496742116800192,
               'Metric precision_score result ': 0.9572122882956067},
           5: {'Metric accuracy_score result ': 0.9496038732394366,
               'Metric balanced_accuracy_score result ': 0.9491958334859812,
               'Metric f1_score result ': 0.9511283706159827, 'Metric recall_score result ': 0.9491958334859812,
               'Metric precision_score result ': 0.9578789467138233},
           6: {'Metric accuracy_score result ': 0.9443221830985915,
               'Metric balanced_accuracy_score result ': 0.9440974427720261,
               'Metric f1_score result ': 0.9459844864833863, 'Metric recall_score result ': 0.9440974427720261,
               'Metric precision_score result ': 0.9537591117982631},
           7: {'Metric accuracy_score result ': 0.9491637323943662,
               'Metric balanced_accuracy_score result ': 0.9492076093350433,
               'Metric f1_score result ': 0.9504387464622264, 'Metric recall_score result ': 0.9492076093350433,
               'Metric precision_score result ': 0.9568871655035718}}
k_fold_results_plots(results=results, "Plots/", "15s_14.5overlap")
