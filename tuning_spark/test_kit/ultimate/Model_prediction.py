# use the ml model to predict the quality of configure

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import *
# from sklearn.neural_network import MLPRegressor  ×
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import  AdaBoostRegressor

# "BaseEnsemble",
# "RandomForestClassifier",
# "RandomForestRegressor",
# "RandomTreesEmbedding",
# "ExtraTreesClassifier",
# "ExtraTreesRegressor",
# "BaggingClassifier",
# "BaggingRegressor",
# "IsolationForest",
# "GradientBoostingClassifier",
# "GradientBoostingRegressor",
# "AdaBoostClassifier",
# "AdaBoostRegressor",
# "VotingClassifier",
# "VotingRegressor",
# "StackingClassifier",
# "StackingRegressor",
# "HistGradientBoostingClassifier",
# "HistGradientBoostingRegressor",

def Choose_parameters_of_RFR(X_train, X_test, y_train, y_test):
    min_error = float('inf')
    filename = './predict_models/Z_finalized_model.sav'
    best_estimators = best_depth = best_features = best_samples = best_bootstrap  = 0
    for n_estimators in [3, 5, 10, 20, 30, 50, 75, 100]:
        for max_depth in [3, 4, 5, 6]:
            for max_features in [1, 3, 5, 7, 9, 12,13,15,16,18]:#20,25,30
                for max_samples in [1, 3, 4, 5, 6, 7]:
                    for best_bootstrap in [True]:
                        model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth,max_features = max_features,max_samples = max_samples, bootstrap = best_bootstrap).fit(X_train, y_train)
                        error = mean_absolute_percentage_error(y_test, model.predict(X_test))
                        if error < min_error:
                            min_error = error; best_estimators = n_estimators; best_depth = max_depth
                            print("\nSetting: best_estimators = {} , best_depth = {} , best_features = {}, best_samples = {}, bootstrap = {}."
                                  .format(n_estimators, max_depth, max_features,max_samples,best_bootstrap))
                            print("the current predict error is {}".format(error))
                            pickle.dump(model, open(filename, 'wb'))

    return

def Choose_parameters_of_GBDT(X_train, X_test, y_train, y_test):
    min_error = float('inf')
    filename = './predict_models/Z_finalized_model.sav'
    best_estimators = best_depth = best_learning_rate = 0
    for n_estimators in [3, 5, 10, 20, 30, 50, 75, 100]:
            for max_depth in [3, 4, 5, 6]:
                for learning_rate in [0.01, 0.1, 1, 10]:
                    model = GradientBoostingRegressor(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate,random_state=43).fit(X_train, y_train)
                    error = mean_absolute_percentage_error(y_test, model.predict(X_test))
                    if error < min_error:
                        min_error = error; best_estimators = n_estimators; best_depth = max_depth; best_learning_rate = learning_rate
                        print("\nSetting: best_estimators = {} , best_depth = {}, best_learning_rate = {}."
                              .format(n_estimators,max_depth,learning_rate))
                        print("the current predict error is {}".format(error))
                        pickle.dump(model, open(filename, 'wb'))
    return


def Choose_parameters_of_BaggingRegressor(X_train, X_test, y_train, y_test):
    min_error = float('inf')
    filename = './predict_models/Z_finalized_model.sav'
    best_estimators = best_features = best_samples = best_bootstrap  = 0
    for n_estimators in [3, 5, 10, 20, 30, 50, 75, 100]:
        for max_features in [1, 3, 5, 7, 9, 11,12]:  #, 13,15,17,19,21
            for max_samples in [1, 3, 4, 5, 6, 7]:
                for best_bootstrap in [True, False]:
                    model = BaggingRegressor(n_estimators = n_estimators, max_features = max_features, max_samples = max_samples,
                                             bootstrap = best_bootstrap).fit(X_train, y_train)
                    error = mean_absolute_percentage_error(y_test, model.predict(X_test))
                    if error < min_error:
                        min_error = error; best_estimators = n_estimators; best_features = max_features; best_samples = max_samples; bootstrap = best_bootstrap
                        print("\nSetting: best_estimators = {} , best_features = {}, best_samples = {}, bootstrap = {}."
                              .format(n_estimators,max_features,max_samples,best_bootstrap))
                        print("the current predict error is {}".format(error))
                        pickle.dump(model, open(filename, 'wb'))
    return



samples_file_path = "/home/hmj/tuning_spark/target/target_spark/data/initial/sql/Z_scan_p_samples_single_150.csv"
Samples = pd.read_csv(samples_file_path, header = 0, encoding = "gbk")  #读取建模用的样本数据
tuned_columns =  ['spark_memory_storageFraction', 'spark_memory_fraction', 'spark_shuffle_sort_bypassMergeThreshold', 'jvm_max_tenuringThreshold', 'jvm_survivor_ratio', 'spark_shuffle_spill_compress', 'spark_executor_cores', 'spark_kryoserializer_buffer', 'spark_storage_replication_proactive', 'spark_executor_memory', 'jvm_gc_timeratio', 'spark_speculation', 'jvm_max_gc_pause', 'spark_locality_wait', 'spark_driver_memory', 'spark_default_parallelism', 'jvm_par_gc_thread', 'spark_shuffle_compress']#, 'spark_broadcast_blockSize', 'spark_task_maxFailures', 'spark_rdd_compress', 'spark_scheduler_revive_interval'
#['spark_broadcast_blockSize', 'spark_task_maxFailures', 'spark_rdd_compress', 'spark_memory_fraction', 'spark_executor_cores', 'spark_speculation', 'spark_shuffle_sort_bypassMergeThreshold', 'spark_storage_replication_proactive', 'spark_driver_memory', 'spark_scheduler_revive_interval', 'jvm_max_gc_pause'] #
# ['datasize','jvm_new_ratio', 'spark_executor_cores', 'spark_memory_offHeap_size', 'spark_storage_replication_proactive',
#                    'spark_shuffle_compress', 'spark_task_maxFailures', 'spark_locality_wait', 'spark_kryoserializer_buffer',
#                    'jvm_survivor_ratio', 'spark_broadcast_blockSize', 'jvm_gc_timeratio', 'spark_driver_memory', 'jvm_gc_collect',
#                    'spark_memory_storageFraction', 'spark_shuffle_sort_bypassMergeThreshold', 'spark_scheduler_revive_interval',
#                    'spark_speculation', 'spark_storage_memoryMapThreshold', 'spark_executor_memory']
    #['spark_broadcast_blockSize', 'spark_task_maxFailures', 'spark_rdd_compress', 'spark_memory_fraction', 'spark_executor_cores', 'spark_speculation', 'spark_shuffle_sort_bypassMergeThreshold', 'spark_storage_replication_proactive', 'spark_driver_memory', 'spark_scheduler_revive_interval', 'jvm_max_gc_pause']#['spark_storage_replication_proactive', 'spark_executor_memory', 'spark_storage_memoryMapThreshold', 'spark_task_maxFailures', 'spark_driver_memory', 'spark_memory_fraction', 'spark_shuffle_sort_bypassMergeThreshold', 'spark_speculation', 'spark_executor_cores', 'spark_shuffle_file_buffer', 'spark_io_compression_codec', 'jvm_max_tenuringThreshold','spark_memory_offHeap_size'] #
X_samples = Samples[tuned_columns]
# X_samples = Samples.iloc[:, :31]
y_samples = Samples.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X_samples, y_samples,test_size = 0.3, random_state = 43)  # 可以把随机数种子去掉
# Choose_parameters_of_GBDT(X_train, X_test, y_train, y_test)
# Choose_parameters_of_BaggingRegressor(X_train, X_test, y_train, y_test)
Choose_parameters_of_RFR(X_train, X_test, y_train, y_test)
# gbdt_model = GradientBoostingRegressor(n_estimators = best_estimators, max_depth = best_depth, learning_rate = best_learning_rate).fit(X_train, y_train)
# model = BaggingRegressor().fit(X_train, y_train)  #可以把随机数种子去掉

filename = './predict_models/Z_finalized_model.sav'  #模型名称
# pickle.dump(model, open(filename, 'wb'))  #离线存储模型
model = pickle.load(open(filename, 'rb')) # 载入离线模型
y_pred = model.predict(X_test)
error = mean_absolute_percentage_error(y_test, y_pred)
plt.scatter(range(1, 101), y_test[:100], label = "Actual Values", color = "blue", s = 2)
plt.scatter(range(1, 101), y_pred[:100], label = "Predicted Values", color = "orange", s = 2)
plt.ylim(50,200)
plt.xlabel("Sample Index")
plt.ylabel("Target Variable (y)")
plt.legend()
plt.show()

print(f"error: {error}")