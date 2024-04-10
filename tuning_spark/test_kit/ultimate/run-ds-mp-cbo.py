# 实验4：用single dataszie的初始化数据跑其他datasize，并且只用一个causal算法

import ast
import asyncio
import copy
import yaml         # pyYaml == 5.1 ; numpy == 1.23.5 ; causal-learn ;
import sys
import logging
import random
import os
from pathlib import Path
from statistics import mean
from tqdm import tqdm
from lib import parse_cmd,run_playbook,get_default,save_tune_conf,find_exist_task_result,divide_config,_print,parse_result,get_default_narrow
from lib.optimizer import create_optimizer
#
import datetime
import numpy as np
import pandas as pd
from src.causal_mode import *
from src.deel_data import *
from networkx import *  # DiGraph, spring_layout, draw
import matplotlib.pyplot as plt
from causallearn.utils.cit import *
from utils_functions import *
from graphs import *
from CBO import *
os.environ['CASTLE_BACKEND'] = 'mindspore'

## hmj ##
def run_updata_model(CM, samples_df, app_setting, tabu_edges, current_columns,
                         obj_columns, NUM_PATHS, model_name, causal_models, task_id):
    print("===============================")
    print("{} Causal Model start!".format(model_name))
    print("===============================")
    graph_start = time.time()
    # # initialize graph and draw graph
    if model_name == 'pc':# causal_models[0]:
        G, di_edges, bi_edges = run_pc_loop(CM, samples_df, tabu_edges, current_columns,     # cur_G,
                                                        obj_columns, NUM_PATHS)
    elif model_name == 'fci': # causal_models[1]:
        di_edges, bi_edges = run_fci_loop(CM, samples_df, tabu_edges, current_columns,
                                                         obj_columns, NUM_PATHS)    # cur_G,
    else:
        di_edges, bi_edges = run_mmhc_loop(CM,samples_df, tabu_edges, current_columns, obj_columns)  #cur_G,
    # #
    graph_build_time = [round(time.time() - graph_start, 3)]  # build graph time

    loop_graph_edges = di_edges + bi_edges  # graph all edges
    cur_graph_edges = get_resolve_edges(current_columns, loop_graph_edges, obj_columns[0])  # without loop edges
    draw_graph(cur_graph_edges, '{}_{}'.format(model_name, task_id ))  # 保存为图片, 命名为 taskid_modelname
    with open('/home/hmj/tuning_spark/target/target_spark/results/causal_graph/causal_path', 'a+') as f:
        f.writelines(str(model_name) + os.linesep + str(cur_graph_edges) + os.linesep)

    # paths = CM.get_causal_paths(current_columns, cur_graph_edges, obj_columns)
    # 每一个datasize model的所有因果路径,没有去除最终节点不是配置参数的路径
    paths = get_ds_G_paths(CM, cur_graph_edges, current_columns, obj_columns)
    paths = paths[obj_columns[0]]
    ds_G_paths = {}
    ds_G_paths[obj_columns[0]] = []
    G_di_edges = []
    # edges # 得到所有因果路径，重组边
    for path in paths:
        # 去除最后一个节点不是配置参数的路径
        if path[len(path)-1] in app_setting.keys():
            ds_G_paths[obj_columns[0]].append(path)
            for i in range(len(path)):
                if i > 0:
                    if (path[i],path[i-1]) not in G_di_edges:
                        G_di_edges.append((path[i],path[i-1]))

    # construct mixed graph ADMG   # 先取出全部因果路径后再构图
    try:
        cur_G = ADMG(current_columns, di_edges=G_di_edges)
    except:
        print("Construct G Error!")   ##
        print(path)
        print(G_di_edges)
        return

    print("===============================")
    print("{} Causal Model end!".format(model_name))
    print("===============================")

    return cur_G, ds_G_paths, cur_graph_edges, graph_build_time

def get_path_columns(samples_df, level_samples, datasize_G_num, similar_ds_list,
                     ds_G, all_paths,app_setting,current_columns, obj_columns, NUM_PATHS):
    # 已知n个基础datasize的causal model
    paths = []
    # all_path = []
    similar_ds = similar_ds_list[0]
    similar_ds_index = similar_ds_list[1]
    min_distance = similar_ds_list[-1][0]

    # 权值设置与计算
    if min_distance == 0:     # 输入是已有的dataszie，给予最高权值
        same_weight = 0.9
        similar_weight = 0
        other_weight = 0.1
    else:
        if len(level_samples) <= datasize_G_num:  # 调整的是不同datasize,并且是初始化
            same_weight = 0
            similar_weight = 0.7
            other_weight = 0.3
        else:                               # 调整的是不同datasize,并且已运行过
            same_weight = 0.92  # 0.6
            similar_weight = 0.05  # 0.3
            other_weight = 0.03   # 0.1

    deel_paths = []
    for ds_index in range(len(all_paths)):  # 5
        for obj_column in obj_columns:  # 1
            ds_paths = all_paths[ds_index][obj_column]
            for path in ds_paths: # n
                if path[-1] not in app_setting.keys():
                    continue
                if path not in deel_paths:
                    deel_paths.append(path)

    ace = {}
    par_ace = {}
    # 每个因果路径path在每个datasize下的平均因果效应
    print("\n========================================================")
    print("Causal Model PATHS ACE Start!")
    print("========================================================")
    for ds_index in range(0, len(level_samples)):  # 5+
        ds_G_index = 0
        # if ds_index >= datasize_G_num:
        #     ds_G_index = similar_ds_index[0]
        # else:
        #     ds_G_index = ds_index

        for path in deel_paths:
            if path not in all_paths[ds_G_index][obj_columns[0]] and ds_G_index < datasize_G_num:
                ds_G_index += 1
            ace[str(path)] = 0
            for i in range(0, len(path)):
                if i > 0:
                    try:
                        obj = CE(graph=ds_G[ds_G_index], treatment=path[i], outcome=path[0])  # 使用ananke估计平均因果效应
                        ace[str(path)] += obj.compute_effect(level_samples[ds_index], "gformula")  # computing the effect   # level_samples[ds_index]
                    except:
                        print(ds_index,path,i)
                        print(ds_G, ds_G_index)
                        return
            if str(path) not in par_ace.keys():
                par_ace[str(path)] = [ace[str(path)]]    # [[ds_index, ace[str(path)]]]
            else:
                par_ace[str(path)].append(ace[str(path)])   # .append([ds_index, ace[str(path)]])
    print("\n========================================================")
    print("Causal Model PATHS ACE End!")
    print("========================================================")

    # 对于参数重复的因果路径:保留<效应大,路径长>的因果路径
    par_weight_ace = {}
    par_paths = {}
    for key in par_ace.keys():
        first_ace = 0
        second_ace = 0
        other_ace = 0
        path = ast.literal_eval(key)# eval(key)
        key_par = path[-1]  # 键值记录
        ace_values = par_ace[key]   # 取出参数在每一个datasize下的ace
        # for ace in par_ace[key]:
        #     ace_values.append(ace[-1])
        if same_weight == 0:     # 输入为新的datasize并且是第一次计算
            first_ace = 0
        else:
            if similar_weight == 0:  # 输入为已有的datasize
                first_ace = same_weight*ace_values[similar_ds_index[0]]
            else:                  # 输入为新的datasize并且不是第一次计算
                first_ace = same_weight*ace_values[len(level_samples)-1]
        for index in similar_ds_index:
            second_ace += similar_weight*ace_values[index]
        for i in range(0, datasize_G_num):
            if i not in similar_ds_index:
                other_ace += other_weight*ace_values[i]
        weitht_ace = first_ace + second_ace + other_ace
        if key_par not in par_paths.keys():
            par_paths[key_par] = path
            #par_mean_ace[key_par] = sum(ace_values)/len(ace_values)
            par_weight_ace[key_par] = weitht_ace
        else:
            if len(path) > len(par_paths[key_par]):  # 优先保留长路径
                par_paths[key_par] = path
            if weitht_ace > par_weight_ace[key_par]:
                par_weight_ace[key_par] = weitht_ace
            # par_mean_ace[key_par] = (par_mean_ace[key_par] + sum(ace_values))/(len(ace_values) + 1)

    sort_mean_ace = {key: value for key, value in sorted(par_weight_ace.items(), key=lambda item: item[1], reverse=True)}

    # extract tunning columns
    num = 0
    cur_tune_conf = {}
    # half par from before
    for key in sort_mean_ace.keys():
        if num < NUM_PATHS/2:
            if key in app_setting.keys():
                cur_tune_conf[key] = app_setting[key]
                paths.append(par_paths[key])
                num += 1
    # half par from after
    for re_key in reversed(sort_mean_ace.keys()):
        if num<len(sort_mean_ace) and num < NUM_PATHS:
            if re_key in app_setting.keys():
                cur_tune_conf[re_key] = app_setting[re_key]
                paths.append(par_paths[re_key])
                num += 1

    with open('/home/hmj/tuning_spark/target/target_spark/results/causal_graph/causal_path', 'a+') as f:
        f.writelines('Key Causal Path:' + str(paths) + os.linesep + str(cur_tune_conf.keys()) + os.linesep)

    # 组合边后单独构建有向无环图，表示当前datasize的causal model
    G_di_edges = []
    for path in paths:
        for i in range(len(path)):
            if i > 0:
                if (path[i],path[i-1]) not in G_di_edges:
                    G_di_edges.append((path[i],path[i-1]))
    # construct mixed graph ADMG
    G = ADMG(current_columns, di_edges=G_di_edges)  # , bi_edges=bi_edges

    print("--------------------------------------------------------------")
    print("Key Causal Path of the causal graph")
    print(paths)
    print("--------------------------------------------------------------")

    return G, G_di_edges, paths, cur_tune_conf
# # -------------------------------------------------------------------------------------------------------
# #

async def main(test_config,os_setting,app_setting,tune_conf):
  global feature_vector_path
  global runtime_path
  assert test_config.tune_os or test_config.tune_app, 'at least one of tune_app and tune_os should be True'

  # hostsname
  tester, testee, slave1, slave2 = test_config.hosts.master,test_config.hosts.master,test_config.hosts.slave1,test_config.hosts.slave2
  # log and resave config,performance
  logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s][%(levelname)s][%(message)s]')
  logger =logging.getLogger('run-causal-cbo')
  handler =logging.FileHandler('run_information/run-causal-cbo.txt')
  handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(message)s]'))
  logger.addHandler(handler)

  ## hmj #causal path nums
  NUM_PATHS = 8
  query = "best"
  input_workload = 'aggregation'  #'bayes'#
  input_datasize = 15000000+1800000 #100000#
  dataszie_values_list = {'wordcount': [800000000, 6400000000],
                   'aggregation': [50000+6000, 100000000+12000000],
                   'bayes': [100000, 2000000]}
  if input_workload not in dataszie_values_list.keys() or \
          input_datasize < dataszie_values_list[input_workload][0] or input_datasize > dataszie_values_list[input_workload][-1]:
      print("\nInput WORKLOAD None! or Inpuut DATASIZE Out of values\n")
      return 0
  ##

  # all parameters
  # tune_configs=[]
  # for key in tune_conf:
  #     tune_configs.append(key)

  ##  hmj  ##
  # columns list
  obj_columns = ['run_times']
  app_columns = app_setting['app_columns']
  os_columns = []     # os_setting['os_columns']  #
  apperf_columns = os_setting['app_perf_columns']
  hdperf_columns = []  # os_setting['hard_perf_columns']  #
  # hdperf_columns = ['migrations', 'context-switches', 'cache-misses', 'branch-misses', 'page-faults', 'cycles', 'instructions', 'L1-dcache-load-misses']

  # use columns
  perf_columns = apperf_columns + hdperf_columns
  tune_columns = app_columns + os_columns
  current_columns = tune_columns + perf_columns + obj_columns
  # get tuning columns values list
  app_setting = get_option_values(app_columns, app_setting['options_vaules'])
  os_setting = get_option_values(os_columns, os_setting['options_vaules'])
  tune_para_setting = {**app_setting, **os_setting}

  # discretize 离散化
  # app_setting = translate_discretize(app_setting)
  # set_values_range = {}
  # for key in app_setting:
  #     set_values_range[key] = app_setting[key]['range']
  ## ##

  # datasize variable
  level_ds_samples_num = 150 #30 #
  datasize_list = {'wordcount': [3200000000],
                   'aggregation': [15000000+1800000],
                   'bayes': [400000]}
  # 计算 Manhattan Distance
  similar_datasize, similar_ds_index, min_distance, similar_ds_sort = get_similar_ds(input_workload, input_datasize, datasize_list)
  similar_ds_list = [similar_datasize] + [similar_ds_index] + [[min_distance]]

  ## hmj ##
  # read initialize data (without os_columns)
  samples_df = pd.read_csv('/home/hmj/tuning_spark/target/target_spark/data/initial/sql/aggregation_samples_single_150.csv')  #aggregation_samples_30
  # extract observer data from csv_file use pandas
  samples_df = samples_df[current_columns]  # save current columns values
  save_columns = current_columns.copy()

  # 根据datasize分层记录samples   # delete invariant variable
  current_columns, ds_samples = get_ds_samples(samples_df.copy(), level_ds_samples_num)
  datasize_num = len(ds_samples)
  samples_df = samples_df[current_columns]
  # 组建因果图的samples数量
  # for index in range(len(ds_samples)):
  #     ds_samples[index] = ds_samples[index].iloc[:ds_samples_num]

  # # normalazation perf_columns
  # normal_df = normal_samples(samples_df, perf_columns)
  ##

  # initialize causal model object
  CM = CausalModel(current_columns)
  # add constrict
  tabu_edges = CM.get_tabu_edges(current_columns, app_columns, obj_columns)

  ## hmj ##        # initialize variable
  apperf_df = pd.DataFrame(columns=apperf_columns)
  hdperf_df = pd.DataFrame(columns=hdperf_columns)
  # 一个所有参数值均为c的列表,“c”代表连续 # 个体因果效应计算
  # var_types = {}
  # for col in current_columns:
  #     var_types[col] = "c"

  # causal model pool type
  causal_models = ['mmhc'] # ['pc', 'fci', 'mmhc']  # , 'ges'
  select_models_list = causal_models.copy()  # 维护模型选择概率列表
  select_BO_models_list = causal_models.copy()[:3]
  first_best_index = False
  select_models = {}   # 记录每次用于推荐配置的因果图模型
  execute_config_opty = []      # 每一次执行的BO model及其实际执行值
  model_execute_config = {}  # 每次选中的每个model的对应配置
  model_execute_opty = {}  # 每次选中的每个model的实际执行值  # 记录不同类型的model当前推荐配置的最优性能目标,
                           # 初始化方法：(1) 初始时都是默认值? (2)初始时都是样本最小值/均值？ (3)初始时每一个model都执行一次cbo,用实际运行值 (4)用cbo的预测值
  G = {}             # 记录每个类型的model因果图
  graph_edges = {}   # 记录每个类型的model所有的节点和边
  graph_build_time = {}  # 每次所选的model的构建时间
  paths = {}         # 记录每个类型的model的因果路径
  current_tune_conf = {}  # 记录每个类型的model调整的参数及其取值范围
  # datasize variable
  ds_G = {}
  ds_graph_edges = {}
  ds_build_time = {}
  all_paths = {}
  # Initialize
  for model_name in causal_models:
      ds_G[model_name] = []
      ds_graph_edges[model_name] = []
      ds_build_time[model_name] = []
      all_paths[model_name] = []

  # CBO variable initialize    # more rater than short
  if len(similar_ds_index) > 2:
      initial_num_obs_samples = level_ds_samples_num * 2    # * len(similar_ds_index)
  else:
      initial_num_obs_samples = level_ds_samples_num * len(similar_ds_index)
  num_interventions = 5
  type_cost = 1
  num_trials = 2  # 50
  causal_prior = True
  num_additional_observations = 20    # 每次每个datasize选2个？num_additional_observations = datasize(num)*2？
  max_N = initial_num_obs_samples + 50
  task = 'min'
  # 根据策略取一定数量的观察数据   # 根据datasize分层选择
  observational_samples, full_observational_samples = get_cbo_samples(ds_samples, similar_ds_sort, initial_num_obs_samples)
  # observational_samples = samples_df[:initial_num_obs_samples]
  # full_observational_samples = samples_df.copy()   # 现假设samples_df中前104个是每个datasize的数据组合而成，后面的数据分为每组为num_additional_observations个
  # if min_distance == 0 or len(similar_ds_index) == 1:  # 输入的是相同datasize or 仅有一个相似的datasize
  #     bo_observe_samples = observational_samples
  #     # current_default_opty = observational_samples.iloc[0, -1]
  # else:
  bo_observe_samples = observational_samples.iloc[0:4]
    # current_default_opty = max([observational_samples.iloc[0, -1], observational_samples.iloc[level_ds_samples_num, -1]])
  initialbo_samples_num = bo_observe_samples.shape[0]

  # CBO variable initial
  record_type_trial = []
  # index = None
  before_dataxlist = []
  before_dataylist = []
  mean_functions_list = []
  var_functions_list = []
  x_dict_mean = {}
  x_dict_var = {}
  model_list = [None] * len(causal_models)  # BO model,对应causal model
  ##  ##

  causal_id = 2
  for task_id in range(causal_id):#tqdm(range(test_config.optimizer.iter_limit)):
    # task_id = 1
    #hmj  # identify and compute causal paths （get config）
    print("\n========================================================")
    print(input_workload, ':', str(task_id) + '_iteration')
    print("========================================================")

    # first , build all graph, initital model pool
    if task_id == 0:
        # 初始时更新所有model
        for j in range(len(causal_models)):
            model_name = causal_models[j]
            # 为每个datasize构建因果图并记录
            for ds_index in range(0, datasize_num):
                print("======  UPDATA CAUSAL MODEL DATASIZE_INDEX: ", ds_index, "======")
                current_G, ds_G_paths, current_edges, current_time = run_updata_model(CM, ds_samples[ds_index], app_setting, tabu_edges, current_columns,
                                                                        obj_columns, NUM_PATHS, model_name, causal_models, task_id)
                ds_G[model_name].append(current_G)
                ds_graph_edges[model_name].append(current_edges)
                ds_build_time[model_name].append(current_time)
                all_paths[model_name].append(ds_G_paths)
                logger.info('the time of build {}_{}  graph is {}'.format(model_name, ds_index, current_time))
            G[model_name], graph_edges[model_name],paths[model_name], current_tune_conf[model_name] = get_path_columns(samples_df, ds_samples.copy(),
                                                                                    datasize_num, similar_ds_list,ds_G[model_name], all_paths[model_name],
                                                                                    app_setting, current_columns, obj_columns, NUM_PATHS)
        # model_name = random.choice(select_models_list)  # 若注释掉，初始默认选择最后一个model
        # select_models[model_name] = [G[model_name], graph_edges[model_name]]
    else:
        # 根据概率表随机选一个model
        model_name = select_models_list[random.randint(0, len(select_models_list) - 1)]  # random.choice(select_models_list)
        new_samples_df = samples_df[level_ds_samples_num * datasize_num + (task_id-1)*cbo_iteration:]
        # 根据相似性更新数据
        if min_distance == 0:  # 输入的是相同datasize
            current_ds_index = similar_ds_index[0]
            if task_id-1 == 0:
                ds_samples[current_ds_index].iloc[0] = new_samples_df.iloc[0].copy()
                new_samples_df = new_samples_df[1:]
            ds_samples[current_ds_index] = pd.concat([ds_samples[current_ds_index],new_samples_df], axis=0, ignore_index=True)
            # 更新所输入的dataszie对应的基础的ds_G
            print("======  UPDATA CAUSAL MODEL DATASIZE_INDEX: ", current_ds_index, "======")
            current_G, ds_G_paths, current_edges, current_time = run_updata_model(CM, ds_samples[current_ds_index], app_setting, tabu_edges, current_columns,
                                                                     obj_columns,NUM_PATHS, model_name, causal_models, task_id)
            ds_G[model_name][current_ds_index] = current_G
            ds_graph_edges[model_name][current_ds_index] = current_edges
            # ds_G_paths = get_ds_G_paths(CM, current_edges, current_columns, obj_columns)
            all_paths[model_name][current_ds_index] = ds_G_paths   # important
            logger.info('the time of build {}_{} graph is {}'.format(model_name, current_ds_index, current_time))
            # select_models_list.append(model_name)
        else:
            if task_id-1 == 0:
                ds_samples.append(new_samples_df)
            else:
                ds_samples[len(ds_samples)-1] = pd.concat([ds_samples[len(ds_samples)-1], new_samples_df], axis=0, ignore_index=True)
        # ds_samples数据已经更新，重新组合该dataszie下的因果图模型和paths
        G[model_name], graph_edges[model_name], paths[model_name], current_tune_conf[model_name] = get_path_columns(samples_df, ds_samples, datasize_num, similar_ds_list,
                                                                                ds_G[model_name], all_paths[model_name],
                                                                                app_setting, current_columns, obj_columns, NUM_PATHS)
    select_models[model_name] = [G[model_name], graph_edges[model_name], paths[model_name]]

    print("\nSelect Update causal model: ", model_name)
    # 去除model调整的相同参数，用于构建BO模型
    tune_configs_list = []
    for CM_name in causal_models:
        tune_configs_list += list(current_tune_conf[CM_name].keys())
    unique_tune_configs = list(set(tune_configs_list))

    #hmj # CBO initialize variables
    update_model_index = causal_models.index(model_name)
    manipulative_variables = tune_columns
    graph = CompleteGraph(bo_observe_samples, causal_models, manipulative_variables, unique_tune_configs, current_tune_conf,
                                                 app_setting, obj_columns)  #  tune_columns --> list(tune_conf.keys())
    # 从所有可干预变量中随机生成N个候选子集  # 候选集从因果图模型中选择
    type_trial = []     # 清除上次model执行记录
    MIS = graph.get_sets()   #current_tune_conf[model_name].keys()   #
    print("Tune Parmaters of causal model In this time:")
    for i in range(len(MIS)):
        print(causal_models[i], ':', MIS[i])
    print("Globel BO Model Unique Initialize Parmaters:  ", unique_tune_configs)

    # Givent the data fit all models used for do calculus   # 初始时只使用4个配置
    try:
        functions = graph.fit_all_models()                    # sometimes input data error
    except:
        print('Input Data Error!')
        unique_tune_configs.append(obj_columns[0])
        print(bo_observe_samples[unique_tune_configs])
        return
    # 定义可干预变量的取值范围
    dict_ranges = graph.get_interventional_ranges()
    # alpha_coverage, hull_obs, coverage_total = compute_coverage(observational_samples, Manipulative_variables, dict_ranges)
    # alpha_coverage, hull_obs, coverage_total = 1, 2, 3     # 用于观察-干预权衡的greedy策略都没用到，可以改成 预测-实际的相对误差 或 实际-最小值/默认值的目标改进
    # 获取干预数据  #第二次更新causal model时，未更新的model不改变data_x_list, data_y_list变量的值    # 改：想办法后续根据实际运行进行更新
    data_x_list, data_y_list, best_intervention_value, opt_y, best_variable, samples_mean_opty = define_initial_data_CBO(observational_samples.copy(),
                                                                                before_dataxlist, before_dataylist, task_id, update_model_index,
                                                                                num_interventions, unique_tune_configs, causal_models, MIS, task, name_index=0)
    print("Samples mean opty:", samples_mean_opty)
    ##
    # current_opt_y = opt_y
    # 修改后成本一致
    costs = graph.get_cost_structure(type_cost=type_cost, Manipulative_variables=manipulative_variables)

    # cbo iteration #
    cbo_iteration = 20   #  3  # len(causal_models) *
    for bo_id in range(cbo_iteration):
        if task_id == 0 and bo_id == 0:
            # 初始时执行默认配置
            sampled_config = get_default(app_setting)
        else:
            print("\n======================================================")
            print("CBO RECOMMENDATED!")
            print("========================================================")
            # hmj # CBO # 依概率选择causal 和 bo model
            (current_best_x, current_best_y, model_list, mean_functions_list, var_functions_list, x_dict_mean, x_dict_var,
             global_opt, total_time, index) = CBO(type_trial, update_model_index, unique_tune_configs, causal_models, model_list,
                                                     select_BO_models_list, mean_functions_list, var_functions_list, x_dict_mean, x_dict_var,
                                                     num_trials, MIS, manipulative_variables,
                                                     data_x_list, data_y_list, best_intervention_value, opt_y, best_variable,
                                                     dict_ranges, functions, bo_observe_samples , initialbo_samples_num, # coverage_total,       # observational_samples
                                                     graph, num_additional_observations, costs, full_observational_samples, task, max_N,
                                                     initial_num_obs_samples, num_interventions, Causal_prior=causal_prior)

            # 将CBO推荐的配置写入
            sampled_config = get_default(app_setting)
            # ref_index = ds_samples[similar_ds_index[0]][[obj_columns[0]]].idxmin()
            # ref_df = ds_samples[similar_ds_index[0]].loc[ref_index]
            # ref = ref_df
            # sampled_config = ref[app_columns].copy().to_dict(orient='records')[0]  # 当前最小性能目标配置
            #
            set_columns = MIS[index]
            set_values = list(current_best_x.values())[index][-1]
            # set tunning paratemer values
            for i in range(len(set_columns)):
                if app_setting[set_columns[i]].get('float') is True:
                    sampled_config[set_columns[i]] = set_values[i]
                    continue
                sampled_config[set_columns[i]] = round(set_values[i])
            ##

        # translate configs
        trans_config = sampled_config.copy()
        trans_config = translate_configs(app_setting, trans_config, 0)  # 0: only trans num type
        sampled_config = translate_configs(app_setting, sampled_config, 1)  # 1: tans num type and vaule
        # print("Current set values:\n", sampled_config)

        if task_id != 0 or bo_id != 0:
            # record config
            save_values = []
            for column in set_columns:
                save_values.append(trans_config[column])
            execute_config_opty.append([causal_models[index], set_columns, save_values])
            try:
                model_execute_config[causal_models[index]].append(save_values)
            except KeyError:
                model_execute_config[causal_models[index]] = [save_values]
            # if bo_id == 0:
            #     default_conf = trans_config

        #test      # setting config and default vaule
        confs = save_tune_conf(task_id*cbo_iteration+bo_id, tune_columns, sampled_config)  # print the tuning parameters value
        save_confs = save_tune_conf(task_id*cbo_iteration+bo_id, tune_columns, trans_config)  # save the tuning parameters value after transform
        sampled_os_config, sampled_app_config = divide_config(sampled_config, os_setting=os_setting, app_setting=app_setting)
        # if tune_app is off, just give sample_app_config a default app_setting value
        if test_config.tune_app is False:
            sampled_app_config = get_default(app_setting)
            sampled_app_config = translate_configs(app_setting, sampled_app_config, 1)

        ## - dump configs ##
        os_config_path = result_dir / f'{task_id*cbo_iteration+bo_id}_os_config.yml'
        os_config_path.write_text(yaml.dump(sampled_os_config, default_flow_style=False))
        app_config_path = result_dir / f'{task_id*cbo_iteration+bo_id}_app_config.yml'
        app_config_path.write_text(yaml.dump(sampled_app_config, default_flow_style=False))

        print("========================================================")
        print("EXECUTE DEFAULT or RECOMMEND CONFIG!")
        print("========================================================")
        start = time.time()
        result_list = []
        skip = False
        # print(default_conf)
        for rep in range(test_config.optimizer.repitition):
            await single_test(                 # add collection performance parameters vaule when runing
                task_name=test_config.task_name,task_id=task_id*cbo_iteration+bo_id,rep=rep,tester=tester,testee=testee,slave1=slave1,slave2=slave2,
                tune_os=(task_id != 0 and test_config.tune_os),clients=test_config.clients,_skip=skip
            )

            _print(f'{task_id*cbo_iteration+bo_id} - {rep}: parsing result...')
            result = parse_result(
                tester_name=test_config.tester,result_dir=result_dir,task_id=task_id*cbo_iteration+bo_id,rep=rep,printer=_print
            )

            result_list.append(- result)
            # result_list.append(round(random.uniform(-160, -500), 3))
            _print(f'{task_id*cbo_iteration+bo_id} - {rep}: done.')

            # hmj # collection one samples
            apperf_filename = '/home/hmj/tuning_spark/target/target_spark/results/temp_feature_vector'
            hdperf_filename = '/home/hmj/cur'    ## _test
            apperf_one_df = get_log_events_samples(apperf_filename, apperf_columns)  #
            apperf_df = pd.concat([apperf_df, apperf_one_df], axis=0)
            hdperf_one_df = get_one_sample(hdperf_filename, hdperf_columns)  #
            hdperf_df = pd.concat([hdperf_df, hdperf_one_df], axis=0)
            #

        # add observation sample        # add sample / updata casual graph
        save_index = sorted(range(len(result_list)), key=lambda k: result_list[k], reverse=True)[1:-1]  # 去掉最小最大值的runtime和events
        result_list = [result_list[i] for i in save_index]
        if len(apperf_columns) !=0:
            apperf_df = apperf_df.iloc[save_index]
        if len(hdperf_columns) !=0:
            hdperf_df = hdperf_df.iloc[save_index]
        metric_result = mean(result_list) if len(result_list) > 0 else .0   # round(random.uniform(-110, -200), 3)  #
        y_new = -metric_result  #
        print("Default or Recommend Config execute opty:", [y_new])

        if task_id == 0 and bo_id == 0:
            current_default_opty = y_new
        else:
            # hmj #CBO add new config and update model
            # set_values_array = np.array([np.round(set_values).astype(int)])
            set_values_array = np.array([save_values])
            # add new outcome
            current_best_y[causal_models[index]].append(y_new)
            execute_config_opty[task_id*cbo_iteration+bo_id-1].append(y_new)
            before_execute_opty = model_execute_opty.copy()
            try:
                model_execute_opty[causal_models[index]].append(y_new)
            except KeyError:
                model_execute_opty[causal_models[index]] = [y_new]
            y_new = np.array([[y_new]])
            data_x, data_y_x = add_data([data_x_list[index], data_y_list[index]], [set_values_array, y_new])
            data_x_list[index] = np.vstack((data_x_list[index], set_values_array))
            data_y_list[index] = np.vstack((data_y_list[index], y_new))
            # 改：opty只在实际执行中选择
            best_intervention_value, opt_y, best_variable = get_current_opty(model_execute_config, model_execute_opty, causal_models, samples_mean_opty, task)
            ## 以什么策略以及什么时候添加model概率
            import_rate = ((-metric_result)-current_default_opty)/current_default_opty
            if causal_models[index] in before_execute_opty.keys() and (-metric_result) <= min(before_execute_opty[causal_models[index]]) and import_rate < -0.2:
                # 添加相对对于自己有改进的           # 并且对于默认配置有改进的 # and (-metric_result)-observational_samples.iloc[0, -1] < -0.2
                select_BO_models_list.append(causal_models[index])     ##
                print("Add BO model in list: ", causal_models[index])
            if set(causal_models) == set(model_execute_opty.keys()) and not first_best_index:
                # 只添加一次比较优秀的
                select_BO_models_list.append(best_variable)     ##
                print("Add B_BO model in list: ", best_variable)
                first_best_index = True
            # Optimise BO model given the new data    # 而 globel model在一次causal迭代后一次性更新
            try:
                model_list[index].set_data(data_x, data_y_x)         # data_x和data_y_x 容易报不是正定矩阵error
            except:
                print(data_x)
                print(data_y_x)
                return
            model_list[index].optimize()
            print("Optimize model end:", index)
            print("Current best model and config:\n", best_variable, MIS[causal_models.index(best_variable)], best_intervention_value, opt_y)
            ##
        # optimizer.add_observation((sampled_config_numeric, metric_result))
        # if hasattr(optimizer, 'dump_state'):
        #     optimizer.dump_state(result_dir / f'{task_id*15+bo_id}_optimizer_state')

        # hmj
        # caculate perf_columns means values of one_configs      # of perf columns
        apperf_df = pd.DataFrame(apperf_df.mean()).transpose()
        hdperf_df = pd.DataFrame(hdperf_df.mean()).transpose()
        # save origination data
        save_configs = save_confs  # current recommand config
        save_configs.extend(apperf_df.iloc[0].values.tolist())
        save_configs.extend(hdperf_df.iloc[0].values.tolist())
        save_configs.append(- metric_result)
        save_configs = pd.DataFrame([save_configs])
        save_configs.columns = save_columns  # 未去除重复性值的columns的所需保存参数
        save_configs = save_configs.round(3)
        # 保留去除值重复的中间变量后的current columns的values
        iteration_configs = save_configs[current_columns]
        samples_df = pd.concat([samples_df, iteration_configs], axis=0, ignore_index=True)   # build causal samples updata
        #
        if task_id == 0 and bo_id == 0:
            observational_samples.iloc[0] = iteration_configs.iloc[0]
            bo_observe_samples.iloc[0] = iteration_configs.iloc[0]
        else:
            observational_samples = pd.concat([observational_samples, iteration_configs], axis=0, ignore_index=True)  # CBO 干预 sample update
            bo_observe_samples = pd.concat([bo_observe_samples, iteration_configs], axis=0, ignore_index=True)   # Globel BO model update
        # 存储未去除值重复的中间变量的数据
        save_configs.to_csv('/home/hmj/tuning_spark/target/target_spark/data/running/sql/aggregation_causal_cbo_samples.csv', mode='a', header=False, index=False)
        # clean temp data
        apperf_df = pd.DataFrame(columns=apperf_columns)
        hdperf_df = pd.DataFrame(columns=hdperf_columns)
        ##

        end = time.time() - start
        logger.info('the time of evaluate configs {} is {}'.format(int(task_id*cbo_iteration+bo_id), round(end)))
        logger.info('the result of task_id {} is {}'.format(int(task_id*cbo_iteration+bo_id), metric_result))
        logger.info('the config of task_id {} is {}'.format(int(task_id*cbo_iteration+bo_id), confs))

    # 根据预期改进选择model,增加相应概率
    # if 所选model在cbo_iteration次中的执行时间最小值小于各model中的最小值,则该model的概率增加,否则列表中执行时间最小值的model的概率增加
    select_models_list.append(best_variable)
    before_dataxlist = data_x_list.copy()
    before_dataylist = data_y_list.copy()
    record_type_trial.append(type_trial)
    print("Current Select Causal BO Model list:", select_BO_models_list)
    print("Current Select Causal Model list", select_models_list)

async def single_test(task_name, task_id, rep, tester, testee, slave1,slave2,tune_os, clients, _skip=False):
  global deploy_spark_playbook_path
  global deploy_hadoop_playbook_path
  global tester_playbook_path
  global osconfig_playbook_path
  global clean_playbook_path

  # for debugging...
  if _skip:
    return

  _print(f'{task_id}: carrying out #{rep} repetition test...')
  try:
  #
    #   - deploy db
    if task_id == 0 and rep == 0:
      _print(f'{task_id} - {rep}: spark_master first deploying...')
      stdout, stderr = await run_playbook(
        deploy_spark_playbook_path,
        host=tester,
        task_name=task_name,
        task_id=task_id,
        task_rep=rep,
      )
      _print(f'{task_id} - {rep}: spark_master first done.')
      # #
      _print(f'{task_id} - {rep}: spark_slave1 first deploying...')
      stdout, stderr = await run_playbook(
        deploy_spark_playbook_path,
        host=slave1,
        task_name=task_name,
        task_id=task_id,
        task_rep=rep,
      )
      _print(f'{task_id} - {rep}: spark_slave1 first done.')

      _print(f'{task_id} - {rep}: spark_slave2 first deploying...')
      stdout, stderr = await run_playbook(
          deploy_spark_playbook_path,
          host=slave2,
          task_name=task_name,
          task_id=task_id,
          task_rep=rep,
      )
      _print(f'{task_id} - {rep}: spark_slave2 first done.')
      #
      # deploy hadoop
      _print(f'{task_id} - {rep}: hadoop_master first deploying...')
      stdout_hadoop, stderr_hadoop = await run_playbook(
            deploy_hadoop_playbook_path,
            host=tester,
            task_name=task_name,
            task_id=task_id,
            task_rep=rep,
        )
      _print(f'{task_id} - {rep}: hadoop_master first done.')

      _print(f'{task_id} - {rep}: hadoop_slave1 first deploying...')
      stdout_hadoop, stderr_hadoop = await run_playbook(
        deploy_hadoop_playbook_path,
        host=slave1,
        task_name=task_name,
        task_id=task_id,
        task_rep=rep,
      )
      _print(f'{task_id} - {rep}: hadoop_slave1 first done.')

      _print(f'{task_id} - {rep}: hadoop_slave2 first deploying...')
      stdout_hadoop, stderr_hadoop = await run_playbook(
        deploy_hadoop_playbook_path,
        host=slave2,
        task_name=task_name,
        task_id=task_id,
        task_rep=rep,
      )
      _print(f'{task_id} - {rep}: hadoop_slave2 first done.')


    if tune_os:
      # os parameters need to be changed
      _print(f'{task_id} - {rep}: setting os parameters...')
      await run_playbook(
          osconfig_playbook_path,
          host=tester,
          task_name=task_name,
          task_id=task_id,
      )
    else:
      # - no need to change, for default testing or os test is configured to be OFF
      _print(
          f'{task_id} - {rep}: resetting os  parameters...')
      await run_playbook(
          osconfig_playbook_path,
          host=tester,
          task_name=task_name,
          task_id=task_id,
          tags='cleanup'
      )
    _print(f'{task_id} - {rep}: done.')

    # - launch test and fetch result
    _print(f'{task_id} - {rep}: hibench testing...')
    await run_playbook(tester_playbook_path,host=testee,target=tester,task_name=task_name,
        task_id=task_id,task_rep=rep,workload_path=str(workload_path),n_client=clients
    )
    _print(f'{task_id} - {rep}: hibench done.')

    _print(f'{task_id} - {rep}: clean logs...')
    await run_playbook(clean_playbook_path,host=testee,target=tester,task_name=task_name,
        task_id=task_id,task_rep=rep,workload_path=str(workload_path),n_client=clients
    )
    _print(f'{task_id} - {rep}: clean logs done.')

    # - cleanup os config
    _print(f'{task_id} - {rep}: cleaning up os config...')
    await run_playbook(
        osconfig_playbook_path,
        host=tester,
        tags='cleanup'
    )
    _print(f'{task_id} - {rep}: done.')
  except RuntimeError as e:
    errlog_path = result_dir / f'{task_id}_error_{rep}.log'
    errlog_path.write_text(str(e))
    print(e)
# -------------------------------------------------------------------------------------------------------
#

run_info='bo.yml'
test_config = parse_cmd(run_info)
assert test_config is not None

# calculate paths
proj_root = Path(__file__, '../../..').resolve()

db_dir = proj_root / f'target/{test_config.target}'
result_dir = db_dir / f'results/{test_config.task_name}'

setting_path = proj_root / \
    f'target/{test_config.target}/os_configs_info.yml'
deploy_spark_playbook_path = db_dir / 'playbook/deploy_spark.yml'
deploy_hadoop_playbook_path = db_dir / 'playbook/deploy_hadoop.yml'
tester_playbook_path = db_dir / 'playbook/tester.yml'
clean_playbook_path = db_dir / 'playbook/clean.yml'
osconfig_playbook_path = db_dir / 'playbook/set_os.yml'
reboot_playbook_path = db_dir / 'playbook/reboot.yml'
workload_path = db_dir / f'workload/{test_config.workload}'

# os_setting_path = proj_root / \
#     f'target/{test_config.target}/os_configs_info.yml'
# app_setting_path = proj_root / \
#     f'target/{test_config.target}/app_configs_info.yml'
# tune_conf_path = proj_root / \
#     f'target/{test_config.target}/low_app_configs_info.yml'
new_os_setting_path = proj_root / f'target/{test_config.target}/new_os_configs_info.yml'
new_app_setting_path = proj_root / f'target/{test_config.target}/app_configs_info.yml'

feature_vector_path = db_dir/ 'results/temp_feature_vector'
runtime_path = db_dir/ 'results/temp_runtime'

# check existing results, find minimum available task_id
# exist_task_id = find_exist_task_result(result_dir)
# if exist_task_id is not None:
#   _print(f'previous results found, with max task_id={exist_task_id}')
#   policy = test_config.exist
#   if policy == 'delete':
#     for file in sorted(result_dir.glob('*')):
#       file.unlink()  #delete a name from filesystem
#     _print('all deleted')
#   elif policy == 'continue':
#     _print(f'continue with task_id={exist_task_id + 1}')
#     init_id = exist_task_id
#   else:
#     _print('set \'exist\' to \'delete\' or \'continue\' to specify what to do, exiting...')
#     sys.exit(0)

# create dirs
result_dir.mkdir(parents=True, exist_ok=True)

# dump test configs
(result_dir / 'test_config.yml').write_text(
    yaml.dump(test_config, default_flow_style=False)
)
_print('test_config.yml dumped')

# read parameters for tuning
os_setting = yaml.load(new_os_setting_path.read_text(), Loader=yaml.FullLoader)
app_setting = yaml.load(new_app_setting_path.read_text(), Loader=yaml.FullLoader)
tune_conf = app_setting['options_vaules']
# os_setting = yaml.load(os_setting_path.read_text(), Loader=yaml.FullLoader)  # pylint: disable=E1101
# app_setting = yaml.load(app_setting_path.read_text(), Loader=yaml.FullLoader)  # pylint: disable=E1101
# tune_conf = yaml.load(tune_conf_path.read_text(), Loader=yaml.FullLoader)

#event loop, main() is async
loop = asyncio.get_event_loop()
loop.run_until_complete(
    main(
        test_config=test_config,
        os_setting=os_setting,
        app_setting=app_setting,
        tune_conf=tune_conf
    )
)
loop.close()

