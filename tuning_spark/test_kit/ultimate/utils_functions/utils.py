## Import basic packages
import random

import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns

## Import emukit function
import emukit
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.acquisition import Acquisition
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.core.optimization import GradientAcquisitionOptimizer

import GPy
from GPy.kern import * # RBF
from GPy.models.gp_regression import GPRegression

from .cost_functions import *
from .causal_acquisition_functions import CausalExpectedImprovement
from .causal_optimizer import CausalGradientAcquisitionOptimizer


def get_new_dict_x(x_new, intervention_variables):
    x_new_dict = {}

    for i in range(len(intervention_variables)):
      x_new_dict[intervention_variables[i]] = x_new[0, i]
    return x_new_dict


def list_interventional_ranges(dict_ranges, intervention_variables):
    list_min_ranges = []
    list_max_ranges = []
    for j in range(len(intervention_variables)):
      list_min_ranges.append(dict_ranges[intervention_variables[j]][0])
      list_max_ranges.append(dict_ranges[intervention_variables[j]][1])
    return list_min_ranges, list_max_ranges


def get_interventional_dict(intervention_variables):
    interventional_dict = {}
    for i in range(len(intervention_variables)):
      interventional_dict[intervention_variables[i]] = ''
    return interventional_dict


def initialise_dicts(causal_models, exploration_set, task):
    current_best_x = {}
    current_best_y = {}
    x_dict_mean = {}
    x_dict_var = {}
    dict_interventions = []

    for i in range(len(exploration_set)):
      variables = causal_models[i]  # "gp_{}".format(i)
      ## This is creating a list of strings 
      dict_interventions.append(variables)


      current_best_x[variables] = []
      current_best_y[variables] = []

      x_dict_mean[variables] = {}
      x_dict_var[variables] = {}

      ## Assign initial values
      if task == 'min':
        current_best_y[variables].append(np.inf)
        current_best_x[variables].append(np.inf)
      else:
        current_best_y[variables].append(-np.inf)
        current_best_x[variables].append(-np.inf)
      
    return current_best_x, current_best_y, dict_interventions  # x_dict_mean, x_dict_var,


def add_data(original, new):
    data_x = np.append(original[0], new[0], axis=0)
    data_y = np.append(original[1], new[1], axis=0)
    return data_x, data_y

## hmj # 找到当前最小的执行时间、配置和model
def get_current_opty(model_execute_config, model_execute_opty, causal_models, samples_mean_opty, task):
    model_best_config = []
    model_best_opty = []
    # 找到各自model的最优执行值和配置，若未实际运行则用样本最小值
    for name in causal_models:
        if name not in model_execute_opty.keys():
            model_best_opty.append(samples_mean_opty[name][-1][0])   # samples_mean_opty[name][-1][0]
            model_best_config.append(samples_mean_opty[name][0])
        else:
            if task == 'min':  #
                best_opty = min(model_execute_opty[name])
                best_opty_index = model_execute_opty[name].index(best_opty)
            else:
                best_opty = max(model_execute_opty[name])
                best_opty_index = model_execute_opty[name].index(best_opty)
            model_best_opty.append(best_opty)
            model_best_config.append(model_execute_config[name][best_opty_index])

    # 找到全局最优执行值
    if task == 'min':
        opt_y = min(model_best_opty)
        global_best_index = model_best_opty.index(opt_y)
    else:
        opt_y = max(model_best_opty)
        global_best_index = model_best_opty.index(opt_y)
    best_intervention_value = model_best_config[global_best_index]
    best_variable = causal_models[global_best_index]

    return best_intervention_value, opt_y, best_variable

## hmj #
def get_bo_model(select_BO_models_list, y_acquisition_list,exploration_set, causal_models):
    index_list = [i[0] for i in sorted(enumerate(y_acquisition_list), key=lambda x:x[1], reverse=True)]  #.shape[0]
    max_vaules_index = index_list[:3]
    max_model_list = []
    # 保留原先位置顺序
    for i in range(len(max_vaules_index)):
        max_model_list.append(causal_models[max_vaules_index[i]])
    j = 0
    for model in causal_models:
        if model in max_model_list:
            select_BO_models_list[j] = model
            j += 1
    ##
    select_model = select_BO_models_list[random.randint(0, len(select_BO_models_list)*10-1) % len(select_BO_models_list)]  # 增大随机范围
    index = causal_models.index(select_model)
    var_to_intervene = exploration_set[index]
    # index_list = np.argsort(np.array(y_acquisition_list))
    return index, var_to_intervene, select_model

#在每次迭代中根据当前的值字典current_y找到最优的变量和值，它根据任务类型计算每个变量的最小值或最大值，并返回最优值
def find_current_global(current_y, dict_interventions, task):
    ## This function finds the optimal value and variable at every iteration
    dict_values = {}
    for j in range(len(dict_interventions)):
        dict_values[dict_interventions[j]] = []

    for variable, value in current_y.items():
        if len(value) > 0:
          if task == 'min':
            dict_values[variable] = np.min(current_y[variable])
          else:
            dict_values[variable] = np.max(current_y[variable])
    if task == 'min':        
      opt_variable = min(dict_values, key=dict_values.get)
    else:
      opt_variable = max(dict_values, key=dict_values.get)
    
    opt_value = dict_values[opt_variable]
    return opt_value
        
def find_next_y_point(space, model, current_global_best, evaluated_set, costs_functions, task = 'min'):
    ## This function optimises the acquisition function and return the next point together with the 
    ## corresponding y value for the acquisition function
    cost_acquisition = Cost(costs_functions, evaluated_set) #创建一个成本对象“cost_acquisition",该对象基于已评估集合和成本函数计算收益函数的成本
    optimizer = CausalGradientAcquisitionOptimizer(space) #创建一个优化器对象”optimizer"，该对象用于优化收益函数
    acquisition = CausalExpectedImprovement(current_global_best, task, model)/cost_acquisition #创建一个收益函数对象，该对象是基于因果期望改进计算的收益函数，除以收益函数的成本
    x_new, _ = optimizer.optimize(acquisition)  #使用优化器对象optimizer对函数acquisition进行优化，得到最优点x_new #ds
    y_acquisition = acquisition.evaluate(x_new)  #使用收益函数对象acquisition在最优点x_new处评估收益函数值，并将其存储在y_acquisition中
    return y_acquisition, x_new    

def caculate_acquisition_vaule(space, model, current_config , current_global_best, evaluated_set, costs_functions, task = 'min'):
    ## This function optimises the acquisition function and return the next point together with the
    ## corresponding y value for the acquisition function
    cost_acquisition = Cost(costs_functions, evaluated_set) #创建一个成本对象“cost_acquisition",该对象基于已评估集合和成本函数计算收益函数的成本
    optimizer = CausalGradientAcquisitionOptimizer(space) #创建一个优化器对象”optimizer"，该对象用于优化收益函数
    acquisition = CausalExpectedImprovement(current_global_best, task, model)/cost_acquisition #创建一个收益函数对象，该对象是基于因果期望改进计算的收益函数，除以收益函数的成本
    # x_new, _ = optimizer.optimize(acquisition)  #使用优化器对象optimizer对函数acquisition进行优化，得到最优点x_new
    y_acquisition = acquisition.evaluate(current_config)  #使用收益函数对象acquisition在最优点x_new处评估收益函数值，并将其存储在y_acquisition中
    return y_acquisition #, x_new

def fit_single_GP_model(X, Y, parameter_list, ard = False):
    kernel = Matern32(X.shape[1], ARD = parameter_list[3], lengthscale=parameter_list[0], variance = parameter_list[1])  # RBF
    gp = GPRegression(X = X, Y = Y, kernel = kernel, noise_var= parameter_list[2])
    # gp.likelihood.variance.fix(1e-2)
    gp.optimize()
    return gp



