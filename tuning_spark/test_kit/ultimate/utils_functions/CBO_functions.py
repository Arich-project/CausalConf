
## Import basic packages
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns



def update_hull(observational_samples, manipulative_variables):
    ## This function computes the coverage of the observations 
    list_variables = []

    for i in range(len(manipulative_variables)):
      list_variables.append(observational_samples[manipulative_variables[i]])

    stack_variables = np.transpose(np.vstack((list_variables)))
    coverage_obs = scipy.spatial.ConvexHull(stack_variables).volume #计算凸包的体积，即覆盖率

    return coverage_obs


def observe(num_observation, complete_dataset = None, initial_num_obs_samples = None):
    observational_samples = complete_dataset[initial_num_obs_samples:(initial_num_obs_samples+num_observation)]
    return observational_samples
    

def compute_coverage(observational_samples, manipulative_variables, dict_ranges):
    list_variables = []
    list_ranges = []

    for i in range(len(manipulative_variables)):
      list_variables.append(observational_samples[manipulative_variables[i]])
      list_ranges.append(dict_ranges[manipulative_variables[i]])

    vertices = list(itertools.product(*[list_ranges[i] for i in range(len(manipulative_variables))]))
    coverage_total = scipy.spatial.ConvexHull(vertices).volume #基于顶点列表"vertices"计算凸包的体积，得到总体覆盖率“coverage_total"

    stack_variables = np.transpose(np.vstack((list_variables)))
    coverage_obs = scipy.spatial.ConvexHull(stack_variables).volume  #计算观测样本的覆盖率”coverage_obs“
    hull_obs = scipy.spatial.ConvexHull(stack_variables) #计算凸包的边界

    alpha_coverage = coverage_obs/coverage_total   #观测样本的覆盖率，即观测样本凸包体积与总凸包体积之比
    return alpha_coverage, hull_obs, coverage_total


def define_initial_data_CBO(interventional_data, before_dataxlist, before_dataylist, task_id, update_model_index,
                            num_interventions, unique_tune_configs, causal_models, exploration_set,  task, name_index):

    data_list = []
    data_x_list = []
    data_y_list = []
    opt_list = []
    value_counts = {}
    max_value_counts = {}
    # samples_best_opty = {}
    samples_mean_opty = {}

    # all_intervente_data = interventional_data.copy()
    # 目标datasize有两个相似时可能会选中相同的，产生数据矩阵不是正定的
    interventional_data = pd.concat([interventional_data[unique_tune_configs], interventional_data.iloc[:, -1]], axis=1)
    for j in range(len(exploration_set)):
      exploration = exploration_set[j]
      ## hmj ## 根据其余参数的综合变化情况从观察数据中选择干预数据
      sort_use_params = list(set(unique_tune_configs).symmetric_difference(set(exploration)))
      filtered_params = sort_use_params.copy()
      filtered_df = interventional_data.copy()
      # sub_data = interventional_data[sort_use_params]
      for i in range(len(filtered_params)):
          for params in sort_use_params:
              value_counts[params] = filtered_df[params].value_counts()          # 剩余的每个params的值的出现次数
              max_value_counts[params] = filtered_df[params].value_counts().max()  # 剩余的每个params出现次数最多的值
          select_params = max(max_value_counts, key=max_value_counts.get)       # 剩余的所有params中值出现次数最多的params
          # print(select_params)
          # print(value_counts[select_params])
          # 该参数取值只有0和1时，保留重复次数最多的值，否则保留值重复次数前3的值
          if len(value_counts[select_params]) < 3:
              max_value = value_counts[select_params].idxmax()
              filtered_df = filtered_df[filtered_df[select_params]==max_value]
          else:
              top_values = value_counts[select_params].index[:3]
              if value_counts[select_params][top_values].sum() > num_interventions:
                  filtered_df=filtered_df[filtered_df[select_params].isin(top_values)]
          sort_use_params = [item for item in sort_use_params if item != select_params]   # 已选的params不参与下一次计数和选择
          value_counts = {}
          max_value_counts = {}
      # 筛选后再次根据重复次数进行保留，因为filtered_df肯定大于num_interventions，所以要保留num_interventions行数据
      for params in filtered_params:
          value_counts[params] = filtered_df[params].value_counts()
          max_value_counts[params] = filtered_df[params].value_counts().max()
      if len(max_value_counts) == 0:   # 该候选集涵盖其他候选集，参数都有
          current_intervente_data = filtered_df
      else:
          select_params = max(max_value_counts, key=max_value_counts.get)
          filtered_df['numcount'] = filtered_df[select_params].map(value_counts[select_params])  # 在剩余的filtered_df中，其他参数的哪个值重复次数最多 .sort_values(ascending=False).index
          sorted_df = filtered_df.sort_values(by='numcount', ascending=False)  # 按参数重复次数最多的那个值排在前面 .loc[sorted_index]
          current_intervente_data = sorted_df[:num_interventions]  # 保留num_interventions行干预数据
          current_intervente_data = current_intervente_data.drop('numcount',axis=1)  # 删除增加的用于计数的列numcount
      # for params in sort_use_params:
      #     value_counts = filtered_df[params].value_counts()            # 依次计算当前探索集中参数重复值的次数
      #     print(params)
      #     print(value_counts)
      #     # 该参数取值只有0和1时，保留重复次数最多的值，否则保留值重复次数前3的值
      #     if len(value_counts) < 3:
      #         max_value = value_counts.idxmax()
      #         filtered_df = filtered_df[filtered_df[params]==max_value]
      #     else:
      #         top_values = value_counts.index[:3]
      #         if value_counts[top_values].sum() > num_interventions:
      #             filtered_df=filtered_df[filtered_df[params].isin(top_values)]
      ##

      data_x = current_intervente_data[exploration]
      data_y = current_intervente_data.iloc[:,-1]

      if len(data_y.shape) == 1:
          data_y = data_y[:,np.newaxis]
      if len(data_x.shape) == 1:
          data_x = data_x[:,np.newaxis]
      all_data = np.concatenate((data_x, data_y), axis=1)

      ## Need to reset the global seed 
      # state = np.random.get_state()
      # np.random.seed(name_index)
      if len(max_value_counts) == 0:
          np.random.shuffle(all_data)
      # np.random.set_state(state)

      subset_all_data = all_data[:num_interventions]

      data_list.append(subset_all_data)
      data_x_list.append(data_list[j][:,:-1])
      data_y_list.append(data_list[j][:,-1][:,np.newaxis])

      # 在第二轮更新causal model时，不更新未选择更新的model
      if task_id >= 1 and j > update_model_index:
          data_x_list[j] = before_dataxlist[j]
          data_y_list[j] = before_dataylist[j]

      if task == 'min':
        opt_list.append(np.min(subset_all_data[:,-1]))
        best_variable = np.where(opt_list == np.min(opt_list))[0][0]
        best_variable = causal_models[best_variable]
        # best_variable = "gp_{}".format(best_variable)
        opt_y = np.min(opt_list)
        opt_intervention_array = data_list[np.where(opt_list == np.min(opt_list))[0][0]]
      else:
        opt_list.append(np.max(subset_all_data[:,-1]))
        best_variable = np.where(opt_list == np.max(opt_list))[0][0]
        # best_variable = "gp_{}".format(best_variable)
        best_variable = causal_models[best_variable]
        opt_y = np.max(opt_list)
        opt_intervention_array = data_list[np.where(opt_list == np.max(opt_list))[0][0]]

    ## hmj # 记录每个model的样本最小值
    # for i in range(len(opt_list)):
    #     config_index = np.where(data_y_list[i] == opt_list[i])[0][0]
    #     samples_best_opty[causal_models[i]] = [list(data_x_list[i][config_index])]
    #     samples_best_opty[causal_models[i]].append([opt_list[i]])
    ##
    ## hmj # 记录每个model的样本均值
    for i in range(len(causal_models)):
        closest_smaller = None
        mean_value = sum(data_y_list[i])/len(data_y_list[i])
        # 返回与均值最接近的更小值
        for value in data_y_list[i]:
            if value < mean_value:
                if closest_smaller is None or mean_value-value < mean_value-closest_smaller:
                    closest_smaller = value
        config_index = np.where(data_y_list[i] == closest_smaller)[0][0]
        samples_mean_opty[causal_models[i]] = [list(data_x_list[i][config_index])]
        samples_mean_opty[causal_models[i]].append(closest_smaller)

    shape_opt = opt_intervention_array.shape[1] - 1
    if task == 'min':
      best_intervention_value = opt_intervention_array[opt_intervention_array[:,shape_opt] == np.min(opt_intervention_array[:,shape_opt]), :shape_opt][0]
    else:
      best_intervention_value = opt_intervention_array[opt_intervention_array[:,shape_opt] == np.max(opt_intervention_array[:,shape_opt]), :shape_opt][0]

    return data_x_list, data_y_list, best_intervention_value, opt_y, best_variable , samples_mean_opty  # samples_best_opty

