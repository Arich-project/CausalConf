# Ours：datasize-aware + model pool + cbo

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
# from utils_functions import *
from graphs import *
from CBO import *
os.environ['CASTLE_BACKEND'] = 'mindspore'

def run_updata_model(CM, samples_df, app_setting, tabu_edges, current_columns,
                         obj_columns, NUM_PATHS, model_name, causal_models, task_id):
    print("===============================")
    print("{} Causal Model start!".format(model_name))
    print("===============================")
    graph_start = time.time()
    if model_name == 'pc':
        G, di_edges, bi_edges = run_pc_loop(CM, samples_df, tabu_edges, current_columns,
                                                        obj_columns, NUM_PATHS)
    elif model_name == 'fci':
        di_edges, bi_edges = run_fci_loop(CM, samples_df, tabu_edges, current_columns,
                                                         obj_columns, NUM_PATHS)
    else:
        di_edges, bi_edges = run_mmhc_loop(CM,samples_df, tabu_edges, current_columns, obj_columns)
    # #
    graph_build_time = [round(time.time() - graph_start, 3)]

    loop_graph_edges = di_edges + bi_edges
    cur_graph_edges = get_resolve_edges(current_columns, loop_graph_edges, obj_columns[0])
    with open('/home/hmj/tuning_spark/target/target_spark/results/causal_graph/causal_path', 'a+') as f:
        f.writelines(str(model_name) + os.linesep + str(cur_graph_edges) + os.linesep)

    paths = get_ds_G_paths(CM, cur_graph_edges, current_columns, obj_columns)
    paths = paths[obj_columns[0]]
    ds_G_paths = {}
    ds_G_paths[obj_columns[0]] = []
    G_di_edges = []
    for path in paths:
        if path[len(path)-1] in app_setting.keys():
            ds_G_paths[obj_columns[0]].append(path)
            for i in range(len(path)):
                if i > 0:
                    if (path[i],path[i-1]) not in G_di_edges:
                        G_di_edges.append((path[i],path[i-1]))

    try:
        cur_G = ADMG(current_columns, di_edges=G_di_edges)
    except:
        print("Construct G Error!")
        print(path)
        print(G_di_edges)
        return

    print("===============================")
    print("{} Causal Model end!".format(model_name))
    print("===============================")

    return cur_G, ds_G_paths, cur_graph_edges, graph_build_time

def get_path_columns(samples_df, level_samples, datasize_G_num, similar_ds_list,
                     ds_G, all_paths,app_setting,current_columns, obj_columns, NUM_PATHS):
    paths = []
    similar_ds = similar_ds_list[0]
    similar_ds_index = similar_ds_list[1]
    min_distance = similar_ds_list[-1][0]

    if min_distance == 0:
        same_weight = 0.9
        similar_weight = 0
        other_weight = 0.1
    else:
        if len(level_samples) <= datasize_G_num:
            same_weight = 0
            similar_weight = 0.7
            other_weight = 0.3
        else:
            same_weight = 0.92
            similar_weight = 0.05
            other_weight = 0.03

    deel_paths = []
    for ds_index in range(len(all_paths)):
        for obj_column in obj_columns:
            ds_paths = all_paths[ds_index][obj_column]
            for path in ds_paths:
                if path[-1] not in app_setting.keys():
                    continue
                if path not in deel_paths:
                    deel_paths.append(path)

    ace = {}
    par_ace = {}
    print("\n========================================================")
    print("Causal Model PATHS ACE Start!")
    print("========================================================")
    for ds_index in range(0, len(level_samples)):
        ds_G_index = 0

        for path in deel_paths:
            if path not in all_paths[ds_G_index][obj_columns[0]] and ds_G_index < datasize_G_num:
                ds_G_index += 1
            ace[str(path)] = 0
            for i in range(0, len(path)):
                if i > 0:
                    try:
                        obj = CE(graph=ds_G[ds_G_index], treatment=path[i], outcome=path[0])
                        ace[str(path)] += obj.compute_effect(level_samples[ds_index], "gformula")
                    except:
                        print(ds_index,path,i)
                        print(ds_G, ds_G_index)
                        return
            if str(path) not in par_ace.keys():
                par_ace[str(path)] = [ace[str(path)]]
            else:
                par_ace[str(path)].append(ace[str(path)])
    print("\n========================================================")
    print("Causal Model PATHS ACE End!")
    print("========================================================")

    par_weight_ace = {}
    par_paths = {}
    for key in par_ace.keys():
        first_ace = 0
        second_ace = 0
        other_ace = 0
        path = ast.literal_eval(key)
        key_par = path[-1]
        ace_values = par_ace[key]
        if same_weight == 0:
            first_ace = 0
        else:
            if similar_weight == 0:
                first_ace = same_weight*ace_values[similar_ds_index[0]]
            else:
                first_ace = same_weight*ace_values[len(level_samples)-1]
        for index in similar_ds_index:
            second_ace += similar_weight*ace_values[index]
        for i in range(0, datasize_G_num):
            if i not in similar_ds_index:
                other_ace += other_weight*ace_values[i]
        weitht_ace = first_ace + second_ace + other_ace
        if key_par not in par_paths.keys():
            par_paths[key_par] = path
            par_weight_ace[key_par] = weitht_ace
        else:
            if len(path) > len(par_paths[key_par]):
                par_paths[key_par] = path
            if weitht_ace > par_weight_ace[key_par]:
                par_weight_ace[key_par] = weitht_ace

    sort_mean_ace = {key: value for key, value in sorted(par_weight_ace.items(), key=lambda item: item[1], reverse=True)}

    num = 0
    cur_tune_conf = {}
    for key in sort_mean_ace.keys():
        if num < NUM_PATHS/2:
            if key in app_setting.keys():
                cur_tune_conf[key] = app_setting[key]
                paths.append(par_paths[key])
                num += 1
    for re_key in reversed(sort_mean_ace.keys()):
        if num<len(sort_mean_ace) and num < NUM_PATHS:
            if re_key in app_setting.keys():
                cur_tune_conf[re_key] = app_setting[re_key]
                paths.append(par_paths[re_key])
                num += 1

    with open('/home/hmj/tuning_spark/target/target_spark/results/causal_graph/causal_path', 'a+') as f:
        f.writelines('Key Causal Path:' + str(paths) + os.linesep + str(cur_tune_conf.keys()) + os.linesep)

    G_di_edges = []
    for path in paths:
        for i in range(len(path)):
            if i > 0:
                if (path[i],path[i-1]) not in G_di_edges:
                    G_di_edges.append((path[i],path[i-1]))
    G_di_edges = get_resolve_edges(current_columns, G_di_edges, obj_columns[0])
    G = ADMG(current_columns, di_edges=G_di_edges)

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

  tester, testee, slave1, slave2 = test_config.hosts.master,test_config.hosts.master,test_config.hosts.slave1,test_config.hosts.slave2
  logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s][%(levelname)s][%(message)s]')
  logger =logging.getLogger('run-causalconf')
  handler =logging.FileHandler('run_information/run-causalconf.txt')
  handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(message)s]'))
  logger.addHandler(handler)

  NUM_PATHS = 8
  # query = "best"
  input_workload = 'bayes' #'join' #'wordcount' #
  input_datasize = 100000 #40000000 + 4800000 #8000000000 #
  dataszie_values_list = {'wordcount': [10000, 32000000000],
                   'join': [100000 + 12000, 100000000 + 12000000],
                   'bayes': [100000, 2000000]}
  if input_workload not in dataszie_values_list.keys() or \
          input_datasize < dataszie_values_list[input_workload][0] or input_datasize > dataszie_values_list[input_workload][-1]:
      print("\nInput WORKLOAD None! or Inpuut DATASIZE Out of values\n")
      return 0

  obj_columns = ['run_times']
  app_columns = app_setting['app_columns']
  os_columns = []
  apperf_columns = os_setting['app_perf_columns']
  hdperf_columns = []

  perf_columns = apperf_columns + hdperf_columns
  tune_columns = app_columns + os_columns
  current_columns = tune_columns + perf_columns + obj_columns
  app_setting = get_option_values(app_columns, app_setting['options_vaules'])
  os_setting = get_option_values(os_columns, os_setting['options_vaules'])
  tune_para_setting = {**app_setting, **os_setting}

  level_ds_samples_num = 30
  datasize_list = {'wordcount': [32000, 3200000, 320000000, 3200000000, 6400000000],
                   'join': [1000000 + 120000, 5000000 + 600000, 10000000 + 1200000, 15000000 + 1800000, 20000000 + 2400000],
                   'bayes': [200000, 300000, 400000, 500000, 600000]}
  similar_datasize, similar_ds_index, min_distance, similar_ds_sort = get_similar_ds(input_workload, input_datasize, datasize_list)
  similar_ds_list = [similar_datasize] + [similar_ds_index] + [[min_distance]]

  all_samples_df = pd.read_csv('/home/hmj/tuning_spark/target/target_spark/data/initial/ml/bayes_samples_30ds.csv')  #
  samples_df = all_samples_df[current_columns].copy()
  save_columns = current_columns.copy()

  current_columns, ds_samples = get_ds_samples(samples_df.copy(), level_ds_samples_num)
  datasize_num = len(ds_samples)
  samples_df = samples_df[current_columns]
  ob_ds_samples = [all_samples_df[['ds']+current_columns].iloc[i:i+level_ds_samples_num] for i in range(0, len(all_samples_df[['ds']+current_columns]), level_ds_samples_num)]

  CM = CausalModel(current_columns)
  tabu_edges = CM.get_tabu_edges(current_columns, app_columns, obj_columns)
  apperf_df = pd.DataFrame(columns=apperf_columns)
  hdperf_df = pd.DataFrame(columns=hdperf_columns)
  causal_models = ['pc','fci','mmhc']
  select_models_list = causal_models.copy()
  select_BO_models_list = causal_models.copy()[:3]
  first_best_index = False
  select_models = {}
  execute_config_opty = []
  model_execute_config = {}
  model_execute_opty = {}
  G = {}
  graph_edges = {}
  graph_build_alltime = 0
  recom_alltime = 0
  paths = {}
  current_tune_conf = {}
  ds_G = {}
  ds_graph_edges = {}
  ds_build_time = {}
  all_paths = {}
  for model_name in causal_models:
      ds_G[model_name] = []
      ds_graph_edges[model_name] = []
      ds_build_time[model_name] = []
      all_paths[model_name] = []

  if len(similar_ds_index) > 2:
      initial_num_obs_samples = level_ds_samples_num * 2
  else:
      initial_num_obs_samples = level_ds_samples_num * len(similar_ds_index)
  type_cost = 1
  num_trials = 2
  causal_prior = True
  num_interventions = 5
  num_additional_observations = 10
  max_N = all_samples_df.shape[0]
  task = 'min'
  record_type_trial = []
  before_dataxlist = []
  before_dataylist = []
  mean_functions_list = []
  var_functions_list = []
  x_dict_mean = {}
  x_dict_var = {}
  model_list = [None] * len(causal_models)
  observational_samples, full_observational_samples = get_cbo_samples(ob_ds_samples, similar_ds_sort, initial_num_obs_samples)
  bo_observe_samples = full_observational_samples.iloc[4::30]
  initialbo_samples_num = bo_observe_samples.shape[0]+1

  causal_id = 2
  for task_id in range(causal_id):
    print("\n========================================================")
    print(input_workload, ':', str(task_id) + '_iteration')
    print("========================================================")
    A_start = time.time()
    if task_id == 0:
        for j in range(len(causal_models)):
            model_name = causal_models[j]
            for ds_index in range(0, datasize_num):
                print("======  UPDATA CAUSAL MODEL DATASIZE_INDEX: ", ds_index, "======")
                current_G, ds_G_paths, current_edges, current_time = run_updata_model(CM, ds_samples[ds_index], app_setting, tabu_edges, current_columns,
                                                                        obj_columns, NUM_PATHS, model_name, causal_models, task_id)
                ds_G[model_name].append(current_G)
                ds_graph_edges[model_name].append(current_edges)
                ds_build_time[model_name].append(current_time)
                all_paths[model_name].append(ds_G_paths)
                draw_graph(current_edges, '{}_{}_{}'.format(model_name, task_id, ds_index))
                logger.info('the time of build {}_{} graph is {}'.format(model_name, ds_index, current_time))
            G[model_name], graph_edges[model_name],paths[model_name], current_tune_conf[model_name] = get_path_columns(samples_df, ds_samples.copy(),
                                                                                             datasize_num, similar_ds_list, ds_G[model_name], all_paths[model_name],
                                                                                             app_setting, current_columns, obj_columns, NUM_PATHS)
    else:
        model_name = select_models_list[random.randint(0, len(select_models_list) - 1)]
        new_samples_df = samples_df[level_ds_samples_num * datasize_num + (task_id-1)*cbo_iteration:]
        if min_distance == 0:
            current_ds_index = similar_ds_index[0]
            if task_id-1 == 0:
                ds_samples[current_ds_index].iloc[0] = new_samples_df.iloc[0].copy()
                new_samples_df = new_samples_df[1:]
            ds_samples[current_ds_index] = pd.concat([ds_samples[current_ds_index],new_samples_df], axis=0, ignore_index=True)
            print("======  UPDATA CAUSAL MODEL DATASIZE_INDEX: ", current_ds_index, "======")
            current_G, ds_G_paths, current_edges, current_time = run_updata_model(CM, ds_samples[current_ds_index], app_setting, tabu_edges, current_columns,
                                                                     obj_columns,NUM_PATHS, model_name, causal_models, task_id)
            ds_G[model_name][current_ds_index] = current_G
            ds_graph_edges[model_name][current_ds_index] = current_edges
            all_paths[model_name][current_ds_index] = ds_G_paths
            draw_graph(current_edges,'{}_{}'.format(model_name, current_ds_index))
            logger.info('the time of build {}_{} graph is {}'.format(model_name, current_ds_index, current_time))
        else:
            if task_id-1 == 0:
                ds_samples.append(new_samples_df)
            else:
                ds_samples[len(ds_samples)-1] = pd.concat([ds_samples[len(ds_samples)-1], new_samples_df], axis=0, ignore_index=True)
        G[model_name], graph_edges[model_name], paths[model_name], current_tune_conf[model_name] = get_path_columns(samples_df, ds_samples.copy(),
                                                                                        datasize_num, similar_ds_list,ds_G[model_name], all_paths[model_name],
                                                                                        app_setting, current_columns, obj_columns, NUM_PATHS)
        logger.info('Select Update causal model: {}'.format(model_name))
    select_models[model_name] = [G[model_name], graph_edges[model_name], paths[model_name]]
    print("\nSelect Update causal model: ", model_name)
    graph_build_alltime = graph_build_alltime + (time.time() - A_start)
    tune_configs_list = []
    for CM_name in causal_models:
        tune_configs_list += list(current_tune_conf[CM_name].keys())
    unique_tune_configs = list(set(tune_configs_list))

    update_model_index = causal_models.index(model_name)
    manipulative_variables = tune_columns
    graph = CompleteGraph(bo_observe_samples, causal_models, manipulative_variables, unique_tune_configs, current_tune_conf,
                                                 app_setting, obj_columns)
    type_trial = []
    MIS = graph.get_sets()
    print("Tune Parmaters of causal model In this time:")
    for i in range(len(MIS)):
        print(causal_models[i], ':', MIS[i])
    print("Globel BO Model Unique Initialize Parmaters:  ", unique_tune_configs)
    try:
        functions = graph.fit_all_models()
    except:
        print('Input Data Error!')
        unique_tune_configs.append(obj_columns[0])
        print(bo_observe_samples[unique_tune_configs])
        return
    dict_ranges = graph.get_interventional_ranges()
    alpha_coverage, hull_obs, coverage_total = 1, 1, 1     # 用于观察-干预权衡的greedy策略
    data_x_list, data_y_list, best_intervention_value, opt_y, best_variable, samples_mean_opty = define_initial_data_CBO(observational_samples.copy(),
                                                                                before_dataxlist, before_dataylist, task_id, update_model_index,
                                                                                num_interventions, unique_tune_configs, causal_models, MIS, task, name_index=0)
    print("Samples mean opty:", samples_mean_opty)
    costs = graph.get_cost_structure(type_cost=type_cost, Manipulative_variables=manipulative_variables)

    cbo_iteration = 20
    for bo_id in range(cbo_iteration):
        B_start = time.time()
        if task_id == 0 and bo_id == 0:
            sampled_config = get_default(app_setting)
        else:
            print("\n======================================================")
            print("CBO RECOMMENDATED!")
            print("========================================================")
            (current_best_x, current_best_y, model_list, mean_functions_list, var_functions_list, x_dict_mean, x_dict_var,
             global_opt, total_time, index) = CBO(type_trial, update_model_index, unique_tune_configs, causal_models, model_list,
                                                     select_BO_models_list, mean_functions_list, var_functions_list, x_dict_mean, x_dict_var,
                                                     num_trials, MIS, manipulative_variables,
                                                     data_x_list, data_y_list, best_intervention_value, opt_y, best_variable,
                                                     dict_ranges, functions, bo_observe_samples , initialbo_samples_num,
                                                     graph, num_additional_observations, costs, full_observational_samples, task, max_N,
                                                     initial_num_obs_samples, num_interventions, Causal_prior=causal_prior)

            sampled_config = get_default(app_setting)
            set_columns = MIS[index]
            set_values = list(current_best_x.values())[index][-1]
            for i in range(len(set_columns)):
                if app_setting[set_columns[i]].get('float') is True:
                    sampled_config[set_columns[i]] = set_values[i]
                    continue
                sampled_config[set_columns[i]] = round(set_values[i])

        trans_config = sampled_config.copy()
        trans_config = translate_configs(app_setting, trans_config, 0)
        sampled_config = translate_configs(app_setting, sampled_config, 1)

        if task_id != 0 or bo_id != 0:
            save_values = []
            for column in set_columns:
                save_values.append(trans_config[column])
            execute_config_opty.append([causal_models[index], set_columns, save_values])
            try:
                model_execute_config[causal_models[index]].append(save_values)
            except KeyError:
                model_execute_config[causal_models[index]] = [save_values]

        confs = save_tune_conf(task_id*cbo_iteration+bo_id, tune_columns, sampled_config)
        save_confs = save_tune_conf(task_id*cbo_iteration+bo_id, tune_columns, trans_config)
        sampled_os_config, sampled_app_config = divide_config(sampled_config, os_setting=os_setting, app_setting=app_setting)
        if test_config.tune_app is False:
            sampled_app_config = get_default(app_setting)
            sampled_app_config = translate_configs(app_setting, sampled_app_config, 1)

        os_config_path = result_dir / f'{task_id*cbo_iteration+bo_id}_os_config.yml'
        os_config_path.write_text(yaml.dump(sampled_os_config, default_flow_style=False))
        app_config_path = result_dir / f'{task_id*cbo_iteration+bo_id}_app_config.yml'
        app_config_path.write_text(yaml.dump(sampled_app_config, default_flow_style=False))

        print("========================================================")
        print("EXECUTE DEFAULT or RECOMMEND CONFIG!")
        print("========================================================")
        C_start = time.time()
        result_list = []
        skip = False
        if task_id == 0 and bo_id == 0:
            iteration_num = test_config.optimizer.repitition
        else:
            iteration_num =1
        for rep in range(iteration_num):
            await single_test(
                task_name=test_config.task_name,task_id=task_id*cbo_iteration+bo_id,rep=rep,tester=tester,testee=testee,slave1=slave1,slave2=slave2,
                tune_os=(task_id != 0 and test_config.tune_os),clients=test_config.clients,_skip=skip
            )

            _print(f'{task_id*cbo_iteration+bo_id} - {rep}: parsing result...')
            result = parse_result(
                tester_name=test_config.tester,result_dir=result_dir,task_id=task_id*cbo_iteration+bo_id,rep=rep,printer=_print
            )

            result_list.append(- result)
            _print(f'{task_id*cbo_iteration+bo_id} - {rep}: done.')

            apperf_filename = '/home/hmj/tuning_spark/target/target_spark/results/temp_feature_vector'
            hdperf_filename = '/home/hmj/cur'
            apperf_one_df = get_log_events_samples(apperf_filename, apperf_columns)
            apperf_df = pd.concat([apperf_df, apperf_one_df], axis=0)
            hdperf_one_df = get_one_sample(hdperf_filename, hdperf_columns)
            hdperf_df = pd.concat([hdperf_df, hdperf_one_df], axis=0)
        C_end = time.time() - C_start

        if task_id == 0 and bo_id == 0:
            save_index = sorted(range(len(result_list)), key=lambda k: result_list[k], reverse=True)[1:-1]
            result_list = [result_list[i] for i in save_index]
        else:
            save_index = [0]
        if len(apperf_columns) !=0:
            apperf_df = apperf_df.iloc[save_index]
        if len(hdperf_columns) !=0:
            hdperf_df = hdperf_df.iloc[save_index]
        metric_result = mean(result_list) if len(result_list) > 0 else .0
        y_new = -metric_result  #
        print("Default or Recommend Config execute opty:", [y_new])

        if task_id == 0 and bo_id == 0:
            current_default_opty = y_new
        else:
            set_values_array = np.array([save_values])
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
            best_intervention_value, opt_y, best_variable = get_current_opty(model_execute_config, model_execute_opty, causal_models, samples_mean_opty, task)
            import_rate = ((-metric_result)-current_default_opty)/current_default_opty
            if causal_models[index] in before_execute_opty.keys() and (-metric_result) <= min(before_execute_opty[causal_models[index]]) and import_rate < -0.2:
                select_BO_models_list.append(causal_models[index])
                print("Add BO model in list: ", causal_models[index])
            if set(causal_models) == set(model_execute_opty.keys()) and not first_best_index:
                select_BO_models_list.append(best_variable)
                print("Add B_BO model in list: ", best_variable)
                first_best_index = True
            try:
                model_list[index].set_data(data_x, data_y_x)
            except Exception as e:
                print(index)
                print(data_x,data_y_x)
                traceback.print_exc()
                return
            model_list[index].optimize()
            print("Optimize model end:", index)
            print("Current best model and config:\n", best_variable, MIS[causal_models.index(best_variable)], best_intervention_value, opt_y)
        B_end = time.time() - B_start
        recom_alltime = recom_alltime + (B_end - C_end)

        apperf_df = pd.DataFrame(apperf_df.mean()).transpose()
        hdperf_df = pd.DataFrame(hdperf_df.mean()).transpose()

        save_configs = save_confs
        save_configs.extend(apperf_df.iloc[0].values.tolist())
        save_configs.extend(hdperf_df.iloc[0].values.tolist())
        save_configs.append(- metric_result)
        save_configs = pd.DataFrame([save_configs])
        save_configs.columns = save_columns
        save_configs = save_configs.round(3)
        iteration_configs = save_configs[current_columns]
        samples_df = pd.concat([samples_df, iteration_configs], axis=0, ignore_index=True)
        iteration_configs.insert(0, 'ds', input_datasize)
        bo_observe_samples = pd.concat([bo_observe_samples, iteration_configs], axis=0, ignore_index=True)
        if task_id == 0 and bo_id == 0:
            observational_samples.iloc[0] = iteration_configs.iloc[0]
        else:
            observational_samples = pd.concat([observational_samples, iteration_configs], axis=0, ignore_index=True)
        save_configs.to_csv('/home/hmj/tuning_spark/target/target_spark/data/running/ml/bayes_causal_cbo_samples.csv', mode='a', header=False, index=False)

        apperf_df = pd.DataFrame(columns=apperf_columns)
        hdperf_df = pd.DataFrame(columns=hdperf_columns)

        logger.info('the time of evaluate configs {} is {}'.format(int(task_id*cbo_iteration+bo_id), round(C_end, 2)))
        logger.info('the result of task_id {} is {}'.format(int(task_id*cbo_iteration+bo_id), metric_result))
        logger.info('the config of task_id {} is {}'.format(int(task_id*cbo_iteration+bo_id), confs))

    select_models_list.append(best_variable)
    before_dataxlist = data_x_list.copy()
    before_dataylist = data_y_list.copy()
    record_type_trial.append(type_trial)
    print("Current Select Causal BO Model list:", select_BO_models_list)
    print("Current Select Causal Model list", select_models_list)
    logger.info("Targe Graph Update Time:{} , All Recommend Time:{}".format(round(graph_build_alltime, 3), round(recom_alltime, 2) ))
    logger.info("Current Select Causal BO Model list:{}".format(select_BO_models_list))
    logger.info("Current Select Causal Model list:{}".format(select_models_list))

async def single_test(task_name, task_id, rep, tester, testee, slave1,slave2,tune_os, clients, _skip=False):
  global deploy_spark_playbook_path
  global deploy_hadoop_playbook_path
  global tester_playbook_path
  global osconfig_playbook_path
  global clean_playbook_path

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
exist_task_id = find_exist_task_result(result_dir)
if exist_task_id is not None:
  _print(f'previous results found, with max task_id={exist_task_id}')
  policy = test_config.exist
  if policy == 'delete':
    for file in sorted(result_dir.glob('*')):
      file.unlink()  #delete a name from filesystem
    _print('all deleted')
  elif policy == 'continue':
    _print(f'continue with task_id={exist_task_id + 1}')
    init_id = exist_task_id
  else:
    _print('set \'exist\' to \'delete\' or \'continue\' to specify what to do, exiting...')
    sys.exit(0)

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
