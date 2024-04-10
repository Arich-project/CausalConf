## Import basic packages
# import sys
# sys.path.append("..")
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from collections import OrderedDict
from matplotlib import cm
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns
import random
from . import graph
from utils_functions import fit_single_GP_model  # cmd run
# from test_kit.ultimate.utils_functions import fit_single_GP_model # pycharm run


from emukit.core.acquisition import Acquisition

## Import GP python packages
import GPy
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression

from .CompleteGraph_DoFunctions import *
from .CompleteGraph_CostFunctions import define_costs

##需要在这里手动定义整个图结构
class CompleteGraph(graph.GraphStructure):
    """
    An instance of the class graph giving the graph structure in the synthetic example 
    
    Parameters
    ----------
    """
    #定义一个pands用来存储初始观察样本，定义一个list，用来存储所有的配置参数
    def __init__(self, observational_samples, causal_models, manipulative_variables,  unique_tune_configs, current_tune_conf, set_values_range, obj_columns):
        self.observational_samples = observational_samples
        all_variables = []
        for column in observational_samples.columns:
            all_variables.append(column)
        self.all_variables = all_variables
        self.manipulative_variables = manipulative_variables   # 所有可变的30个参数
        self.tune_configs = unique_tune_configs                     # 所选用于建一个完整的BO模型的参数
        self.current_tune_conf = current_tune_conf          # 候选集：每一个结构学习算法构建的可调整的参数及其范围
        self.set_values_range = set_values_range            # 所有可调整参数及其范围
        self.causal_model = causal_models
        self.MIS = []
        self.obj_column = obj_columns[0]

    #节点直接的固定因果关系，这个应该用不到，直接注视掉
    def define_SEM(self):

        def fU1(epsilon, **kwargs):
          return epsilon[0]

        def fU2(epsilon, **kwargs):
          return epsilon[1]

        def fF(epsilon, **kwargs):
          return epsilon[8]

        def fA(epsilon, U1, F, **kwargs):
          return F**2 + U1 + epsilon[2]

        def fB(epsilon, U2, **kwargs):
          return U2 + epsilon[3]

        def fC(epsilon, B, **kwargs):
          return np.exp(-B) + epsilon[4]

        def fD(epsilon, C, **kwargs):
          return np.exp(-C)/10. + epsilon[5]

        def fE(epsilon, A, C, **kwargs):
          return np.cos(A) + C/10. + epsilon[6]

        def fY(epsilon, D, E, U1, U2, **kwargs):
          return np.cos(D) - D/5. + np.sin(E) - E/4. + U1 + np.exp(-U2) + epsilon[7]

        graph = OrderedDict ([
              ('U1', fU1),
              ('U2', fU2),
              ('F', fF),
              ('A', fA),
              ('B', fB),
              ('C', fC),
              ('D', fD),
              ('E', fE),
              ('Y', fY),
            ])
        return graph

    # #在初始化里面顶一个两个list，一个是可干预参数集合，一个是候选集合，然后在这个function中返回
    def get_sets(self):
        MIS = []
        for key in self.current_tune_conf.keys():
            MIS.append(list(self.current_tune_conf[key].keys()))
        # M_v_number = len(self.manipulative_variables)
        # for _ in range(int(M_v_number/2)):
        #     temp = random.sample(self.manipulative_variables, int(M_v_number * 2 / 3))
        #     MIS.append(temp)
        self.MIS = MIS
        return MIS

    #返回所有的可干预参数
    def get_set_BO(self):         #能调整的几个参数
        return self.manipulative_variables
    def get_configs(self):
        return self.tune_configs

    #定义可干预参数的取值空间，这个也在__init__function中进行初始化，然后在这个函数中返回就Ok了
    def get_interventional_ranges(self):
        dict_ranges = {}
        for column in self.manipulative_variables:
            if self.set_values_range[column].get('range') is None:
                temp = [self.set_values_range[column]['min'], self.set_values_range[column]['max']]
            else:
                temp = [self.set_values_range[column]['range'][0], self.set_values_range[column]['range'][-1]]
            dict_ranges[column] = temp
        return dict_ranges      #定义每个参数的取值范围

    #拟合函数，__init__function里面的观察样本和候选参数集合，去拟合多个GP模型，然后在这个function中进行返回
    def fit_all_models(self):
        print("\nInitialize global model Start!")
        ## Fit all conditional models
        functions = {}
        parameter_list = [1., 1., 1., False]
        # for i in range(len(self.MIS)):
        #     X = self.observational_samples[self.MIS[i]].values
        #     Y = self.observational_samples[self.obj_column].to_frame().values
        #     function_name = self.causal_model[i]  # "GP_{}".format(i)
        #     functions[function_name] = fit_single_GP_model(X, Y, parameter_list)
        X = self.observational_samples[self.tune_configs].values
        Y = self.observational_samples[self.obj_column].to_frame().values
        function_name = "compute_function"
        functions[function_name] = fit_single_GP_model(X, Y, parameter_list)

        print("Initialize global model end!\n")

        return functions



    #基于新的观察样本，重新拟合一下function就Ok了
    def refit_models(self, observational_samples):
        print("Refit global model start!")
        ## refit all conditional models
        functions = {}
        parameter_list = [1., 1., 10., False]
        # for i in range(len(self.MIS)):
        #     X = observational_samples[self.MIS[i]]
        #     Y = observational_samples[self.obj_column].to_frame()
        #     function_name = self.causal_model[i]  # "GP_{}".format(i)
        #     functions[function_name] = fit_single_GP_model(X, Y, parameter_list)
        X = observational_samples[self.tune_configs]
        Y = observational_samples[self.obj_column].to_frame()
        function_name = "compute_function"
        functions[function_name] = fit_single_GP_model(X, Y, parameter_list)

        print("Refit global model end!")

        return functions

    #这个已经Ok了
    def get_cost_structure(self, type_cost,Manipulative_variables):
        costs = define_costs(type_cost, Manipulative_variables)
        return costs

    #重新手动定义
    def get_all_do(self ):
        do_dict = {}
        do_dict['compute_function'] = compute_do
        return do_dict



