## Import basic packages
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns


def get_do_function_name(intervention_variables):
    # string = ''
    # for i in range(len(intervention_variables)):
    #     string += str(intervention_variables[i])
    # total_string = 'compute_do_' + string
    total_string = 'compute_function'
    return total_string


## Given a do function, this function is computing the mean and variance functions needed for the Causal prior 
def mean_var_do_functions(do_effects_function, observational_samples, functions):
    xi_dict_mean = {}
    def mean_function_do(x):
        num_interventions = x.shape[0]
        mean_do = np.zeros((num_interventions, 1))
        for i in range(num_interventions):
            xi_str = str(x[i])
            if xi_str in xi_dict_mean:
                mean_do[i] = xi_dict_mean[xi_str]
            else:
                mean_do[i], _ = do_effects_function(observational_samples, functions, x[i])
                xi_dict_mean[xi_str] = mean_do[i]
        return np.float64(mean_do)
    
    xi_dict_var = {}
    def var_function_do(x):
        num_interventions = x.shape[0]
        var_do = np.zeros((num_interventions, 1))
        for i in range(num_interventions):
            xi_str = str(x[i])
            if xi_str in xi_dict_var:
                var_do[i] = xi_dict_var[xi_str]
            else:
                _, var_do[i] = do_effects_function(observational_samples, functions, x[i])
                xi_dict_var[xi_str] = var_do[i]
        return np.float64(var_do)

    return mean_function_do, var_function_do


def update_all_do_functions(graph, exploration_set, functions, dict_interventions, observational_samples, x_dict_mean, x_dict_var,
                            all_variables):

    mean_functions_list = []
    var_functions_list = []

    for j in range(len(exploration_set)):
        # 此处可修改functions
        # function = functions[list(functions.keys())[j]]
        if j >= len(functions):
            function = functions[list(functions.keys())[0]]
        else:
            function = functions[list(functions.keys())[j]]
        ##
        mean_functions_list.append(update_mean_fun(graph, function, dict_interventions[j],          # 均值函数
                                                   observational_samples, x_dict_mean, all_variables, exploration_set[j]))
        var_functions_list.append(update_var_fun(graph, function, dict_interventions[j],
                                                 observational_samples, x_dict_var, all_variables, exploration_set[j]))   # 方差函数
    return mean_functions_list, var_functions_list

 # 自定义均值函数
def update_mean_fun(graph, functions, variables, observational_samples, xi_dict_mean, all_variables, part_variable):

    def compute_mean(num_interventions, x, xi_dict_mean, compute_do):
        mean_do = np.zeros((num_interventions, 1))
        for i in range(num_interventions):
            xi_str = str(x[i])
            if xi_str in xi_dict_mean:
                mean_do[i] = xi_dict_mean[xi_str]
            else:
                mean_do[i], _ = compute_do(observational_samples, functions,  x[i], all_variables, part_variable)
                xi_dict_mean[xi_str] = mean_do[i]
        return mean_do


    do_functions = graph.get_all_do()
    function_name = get_do_function_name(variables)

    def mean_function_do(x):
        num_interventions = x.shape[0]
        mean_do = compute_mean(num_interventions, x, xi_dict_mean[variables], do_functions[function_name])
        return np.float64(mean_do)

    return mean_function_do        #

# 自定义方差函数
def update_var_fun(graph, functions, variables, observational_samples, xi_dict_var, all_variables, part_variable):

    def compute_var(num_interventions, x, xi_dict_var, compute_do):
        var_do = np.zeros((num_interventions, 1))
        for i in range(num_interventions):
            xi_str = str(x[i])
            if xi_str in xi_dict_var:
                var_do[i] = xi_dict_var[xi_str]
            else:
                _, var_do[i] = compute_do(observational_samples,functions,x[i],all_variables, part_variable)
                xi_dict_var[xi_str] = var_do[i]

        return var_do

    do_functions = graph.get_all_do()
    function_name = get_do_function_name(variables)

    def var_function_do(x):
        num_interventions = x.shape[0]    
        var_do = compute_var(num_interventions, x, xi_dict_var[variables], do_functions[function_name])
        return np.float64(var_do)

    return var_function_do  