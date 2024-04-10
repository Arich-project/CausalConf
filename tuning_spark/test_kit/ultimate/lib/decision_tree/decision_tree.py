from __future__ import division, print_function
import os
import numpy as np
import warnings
import math
from .decision_tree_model import RegressionTree
from ..optimizer import create_optimizer
from logging import Logger
warnings.filterwarnings("ignore",category=Warning)


def read_history_data():
    # use to read the history tune data
    root_path = '/home/hmj/tuning_spark/target/target_spark/results/history_data/'
    infor_dir = ['tune_config', 'run_time', 'feature_vector']
    history_config_paths = root_path + infor_dir[0]  # 目录的表示需要优化下
    history_runtime_paths = root_path + infor_dir[1]
    history_log_paths = root_path + infor_dir[2]
    sub_dir = ['micro', 'ml', 'sql']
    history_data_x = {}
    history_data_y = {}
    history_data_z = {}
    history_data_s = {}
    for i in range(len(sub_dir)):
        history_config_path = history_config_paths + '/' + sub_dir[i] + '/'
        history_runtime_path = history_runtime_paths + '/' + sub_dir[i] + '/'
        history_log_path = history_log_paths + '/' + sub_dir[i] + '/'
        history_config_files = os.listdir(history_config_path)
        history_runtime_files = os.listdir(history_runtime_path)
        history_log_files = os.listdir(history_log_path)
        for file_name in history_config_files:
            with open(history_config_path + file_name, encoding='utf-8') as f1:
                i = 0
                all_infor = []
                for line in f1.readlines():
                    row_infor = []
                    if i == 0:
                        i += 1
                        continue
                    line = line.split()
                    for j in range(len(line)):
                        row_infor.append(float(line[j]))
                    all_infor.append(row_infor)
            history_data_x[file_name] = np.asarray(all_infor, dtype=object)
        for file_name in history_runtime_files:
            with open(history_runtime_path + file_name, encoding='utf-8') as f2:
                i = 0
                all_infor = []
                for line in f2.readlines():
                    if i == 0:
                        i += 1
                        continue
                    if line == '\n':
                        continue
                    line = line.split()
                    all_infor.append(float(line[4]))
            history_data_y[file_name] = np.asarray(all_infor, dtype=object)
        for file_name in history_log_files:
            with open(history_log_path + file_name, encoding='utf-8') as f3:
                all_infor = []
                stages = []
                for line in f3.readlines():
                    line = line.split()
                    stage_feature = line[0]
                    row_infor = []
                    for i in range(1, len(line)):
                        row_infor.append(float(line[i]))
                    all_infor.append(row_infor)
            history_data_z[file_name] = np.asarray(all_infor)
            history_data_s[file_name] = stage_feature
    return history_data_x, history_data_y, history_data_z,history_data_s

def normalize(y):
    # normalize the value of y compare to default value
    default_value=y[0]
    normalize_y=(y[1:]-default_value)/default_value
    return normalize_y
class SingleModel(object):
    def __init__(self,modelname=None,feature=None,stage=None):
        self.modelname = modelname
        self.feature = feature
        self.stage = stage
        self.y_pred =0
        self.model = None
        self.SimilarityWeight=None
        self.SortErrWeight=None
    def BuildModel(self,config,runtime):
        self.model = RegressionTree()
        self.model.fit(config,runtime)
    def BuildGPModel(self,OptimizerName,tune_conf,extra_vars,config,runtime):
        self.model = create_optimizer(OptimizerName,tune_conf,extra_vars)
        for i in range(len(runtime)):
            self.model.add_observation((config[i],-runtime[i]))
    def getSimilarity(self,x_feature,stage):
        temp = np.concatenate((x_feature, self.feature))
        mean = np.mean(temp, axis=0)
        std = []
        for i in range(temp.shape[1]):
            st = np.std(temp[:, i])
            std.append(st)
        std = np.array(std)
        for i in range(temp.shape[1]):
            if std[i] != 0:
                temp[:, i] = (temp[:, i] - mean[i]) / std[i]
        np_feature=np.asarray(self.feature)
        row = np_feature.shape[0]
        col = np_feature.shape[1]
        sum = 0
        for i in range(row):
            temp = 0
            for j in range(col - 20):
                if j % 4 == 2 or j % 4 == 3:
                    if (x_feature[0][j] + self.feature[i][j]) != 0:
                        temp += pow(abs(x_feature[0][j] - self.feature[i][j]) / (x_feature[0][j] + self.feature[i][j]), 2)
            if stage == self.stage:
                temp += 0
            else:
                temp += 2
            temp = math.sqrt(temp)
            sum = sum + temp
        sum = sum / row
        self.SimilarityWeight = math.pow(sum,-1)

class ModelPool(object):
    def __init__(self,configs=None,runtimes=None,features=None,stage=None):
        self.configs=configs
        self.runtimes=runtimes
        self.features=features
        self.stage=stage
        self.modelpools=[]
        self._logger=Logger('logger')
    def BuildModels(self):
        for key in self.configs:
            x_train=self.configs[key]
            if key in self.runtimes and key in self.features and key in self.stage:
                y_train=self.runtimes[key]
                z_feature=self.features[key]
                s_stage=self.stage[key]
            else :
                self._logger.warning('the runtimes or features of {} is not exist'.format(key))
                continue
            if x_train.shape[0] != y_train[1:].shape[0]:
                self._logger.warning('{}:the number of configuration and corresponding performance is not match'.format(key))
                continue
            y_train=normalize(y_train)
            model=SingleModel(key,z_feature,s_stage)
            model.BuildModel(x_train,y_train)
            self.modelpools.append(model)
        return self.modelpools
    def BuildBOModels(self,OptimizerName,tune_conf,extra_vars):
        for key in self.configs:
            x_train=self.configs[key]
            if key in self.runtimes and key in self.features and key in self.stage:
                y_train=self.runtimes[key]
                z_feature=self.features[key]
                s_stage=self.stage[key]
            else :
                self._logger.warning('the runtimes or features of {} is not exist'.format(key))
                continue
            if x_train.shape[0] != y_train[1:].shape[0]:
                self._logger.warning('{}:the number of configuration and corresponding performance is not match'.format(key))
                continue
            y_train=normalize(y_train)
            #y_train  = (y_train[1:])
            model=SingleModel(key,z_feature,s_stage)
            model.BuildGPModel(OptimizerName,tune_conf,extra_vars,x_train,y_train)
            self.modelpools.append(model)
        return self.modelpools







# x,y,z,j=read_history_data() # read the history_tuning_config
# for key in x:
#     print(key+' :the shape of x is {}'.format(x[key].shape)+'and the shape of y is {}'.format(y[key].shape)+
#           key+' :the shape of z is {}'.format(z[key].shape))
#     # and y[0] is the executor time of default configure
# modelpool=ModelPool(x,y,z).BuildModels()
# x_test=x['terasort-rand-30']
# y_test=y['terasort-rand-30']
# for i in range(len(modelpool)):
#     if modelpool[i].modelname=='terasort-rand-500':
#         print(modelpool[i].modelname)
#         y_pred=modelpool[i].model.predict(x_test)
#         for j in range(len(y_pred)):
#             print("compare the really and predict value:{}-{}".format(y_test[j],y_pred[j]))
# print("the every things is right")




# model = RegressionTree()
# x_train=x['terasort-rand-30']
# y_train=normalize(y['terasort-rand-30'])
# x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, test_size=0.3)
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)
# print('see the value of y_test {}'.format(y_test))
# print('see the value of y_pred {}'.format(y_pred))
# error = mean_squared_error(y_test,y_pred)
# print("the value of error is {}".format(error))




