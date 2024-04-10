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
from lib import parse_cmd,run_playbook,get_default,save_tune_conf,find_exist_task_result,\
    divide_config,_print,parse_result,get_default_narrow
from lib.optimizer import create_optimizer
#
import datetime
import numpy as np
import pandas as pd
from src.causal_mode import *
from src.deel_data import *
from ananke.graphs import ADMG
from networkx import *  # DiGraph, spring_layout, draw
import matplotlib.pyplot as plt
from causallearn.utils.cit import *

# def run_fci_loop(CM, df,
#                      tabu_edges, columns, obj_columns,
#                      NUM_PATHS):
#     """This function is used to run fci in a loop"""
#     # NOTEARS causal model hyperparmas
#     #_, notears_edges = CM.learn_entropy(df, tabu_edges, 0.75)
#     # get bayesian network from DAG obtained by NOTEARS
#     # bn = BayesianNetwork(sm)
#     fci_edges = CM.learn_fci(df, tabu_edges)
#     edges = []
#     # resolve notears_edges and fci_edges and update  #使用熵定向策略消除不确定的边
#     di_edges, bi_edges = CM.resolve_edges(edges, fci_edges, columns,
#                                           tabu_edges, NUM_PATHS, obj_columns)
#     # construct mixed graph ADMG
#     G = ADMG(columns, di_edges=di_edges, bi_edges=bi_edges)
#
#     return G, di_edges, bi_edges
#
# def run_ges_loop(df, tabu_edges, columns, obj_columns):
#     """This function is used to run ges in a loop"""
#
#     print("GES START!")
#     ges_results = ges(np.array(df), score_func='local_score_BDeu')
#     G = ges_results['G']
#     # graph_edges = G.get_graph_edges()
#     # print(G)
#     gmatrix = G.graph
#     # gmatrix = np.array(gmatrix)
#     GM = Graph_Matrix(columns, gmatrix)
#     orig_di_edges, orig_bi_edges, num_graph = GM.getGESEdges()
#
#     di_edges = []
#     bi_edges = []
#     for edges in orig_di_edges:
#         # if edges[0] == obj_columns[0]:
#         #     di_edges.append(edges) # di_edges.append((edges[1], edges[0]))
#         if edges in tabu_edges:
#             continue
#         di_edges.append(edges)
#     for edges in orig_bi_edges:
#         if edges in tabu_edges:
#             continue
#         bi_edges.append(edges)
#
#     return G, di_edges, bi_edges
#
# def run_mmhc_loop(df, tabu_edges, columns, obj_columns):
#     """This function is used to run mmhc in a loop"""
#
#     mmhc_results = MMHC(df, alpha= 0.23, score_function='bic') # , score_function='bic'
#     GM = Graph_Matrix(columns, mmhc_results[0])
#     orig_di_edges, orig_bi_edges, num_graph = GM.getmmhcEdges()
#
#     di_edges = []
#     bi_edges = []
#     for edges in orig_di_edges:
#         # if edges[0] == obj_columns[0]:
#         #     di_edges.append(edges) #di_edges.append((edges[1], edges[0]))
#         if edges in tabu_edges:
#             continue
#         di_edges.append(edges)
#     for edges in orig_bi_edges:
#         if edges in tabu_edges:
#             continue
#         bi_edges.append(edges)
#
#     # construct mixed graph ADMG
#     G = ADMG(columns, di_edges=di_edges, bi_edges=bi_edges)
#
#     return G, di_edges, bi_edges
# # -------------------------------------------------------------------------------------------------------
# #

async def main(test_config,os_setting,app_setting,tune_conf):
  global feature_vector_path
  global runtime_path
  assert test_config.tune_os or test_config.tune_app, 'at least one of tune_app and tune_os should be True'

  # hmj #causal path nums
  NUM_PATHS = 10
  query = "best"
  ##

  # parameters     # initialize and add nodes for causal graph
  tune_configs=[]
  for key in tune_conf:
      tune_configs.append(key)

  #hmj
  # columns list
  obj_columns = ['run_times']
  os_columns = os_setting['os_columns']
  app_columns = app_setting['app_columns']
  hdperf_columns = os_setting['hard_perf_columns']
  apperf_columns = os_setting['app_perf_columns']

  # select columns
  hdperf_columns = []   # hdperf_columns = ['migrations', 'context-switches', 'cache-misses', 'branch-misses', 'page-faults', 'cycles', 'instructions', 'L1-dcache-load-misses']
  os_columns = []

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
  ##

  #hmj # read initialize data (without os_columns)
  samples_df = pd.read_csv('/home/hmj/tuning_spark/target/target_spark/data/initial/micro/terasort_samples_bo.csv')  # extract data from csv_file use pandas
  samples_df = samples_df[current_columns]    # deel with
  # delete invariant variable
  filtered_df = samples_df.copy()
  unique_counts = filtered_df.nunique()
  samples_df = filtered_df.loc[:, unique_counts != 1]
  save_columns = current_columns.copy()
  current_columns = samples_df.columns.tolist()
  # samples_df = samples_df.head(35)

  # normalazation perf_columns
  # normal_df = normal_samples(samples_df, perf_columns)

  # initialize causal model object
  CM = CausalModel(current_columns)
  # add constrict
  tabu_edges = CM.get_tabu_edges(current_columns, app_columns, obj_columns)

  # start build graph
  graph_start = time.time()
  print("Constructed Causal model")
  # fci: build
  G, di_edges, bi_edges = run_fci_loop(CM, samples_df, tabu_edges,
                                           current_columns, obj_columns, NUM_PATHS)
  ##
  # mmhc: build
  # G, di_edges, bi_edges = run_mmhc_loop(samples_df, tabu_edges,
  #                                       current_columns, obj_columns)
  ##
  # pc: build
  # G, di_edges, bi_edges = run_pc_loop(CM, samples_df, tabu_edges,
  #                                          current_columns, obj_columns, NUM_PATHS)
  ##
  # ges: build
  # G, di_edges, bi_edges = run_ges_loop(samples_df, tabu_edges,
  #                                      current_columns, obj_columns)
  ##
  # notears: build
  # G, di_edges, bi_edges = run_notears_loop(samples_df, tabu_edges,
  #                                       current_columns, obj_columns)
  ##
  graph_end = time.time() - graph_start
  with open('/home/hmj/tuning_spark/target/target_spark/results/causal_graph/causal_path', 'a+') as f:
      f.writelines(str(di_edges) + str(bi_edges) + os.linesep)
  # initialize graph and draw graph
  draw_graph(di_edges+bi_edges, 0)
  ##

  # hostsname
  tester, testee, slave1,slave2= test_config.hosts.master,test_config.hosts.master,test_config.hosts.slave1,test_config.hosts.slave2
  # log and resave config,performance
  logging.basicConfig(level=logging.INFO,format='[%(asctime)s][%(name)s][%(levelname)s][%(message)s]')
  logger=logging.getLogger('run-causal-bo')
  handler=logging.FileHandler('run_information/run-causal-bo.txt')
  handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(message)s]'))
  logger.addHandler(handler)
  logger.info('the time of build graph {} is {}'.format(0, graph_end))

  #hmj
  # initialize variable
  apperf_df = pd.DataFrame(columns=apperf_columns)
  hdperf_df = pd.DataFrame(columns=hdperf_columns)
  # 一个所有参数值均为c的列表,“c”代表连续
  # var_types = {}
  # for col in current_columns:
  #     var_types[col] = "c"
  ##

  causal_id = 1
  for task_id in range(causal_id):#tqdm(range(test_config.optimizer.iter_limit)):
    #hmj  # identify and compute causal paths （get config）
    print("========================================================")
    print(task_id)
    print("========================================================")
    # get current_min_configs
    ref_index = samples_df[[obj_columns[0]]].idxmin()
    ref_df = samples_df.loc[ref_index]
    ref = ref_df.iloc[0]
    previous_config = ref[app_columns].copy()  # 当前最小性能目标配置
    # identify causal paths
    paths = CM.get_causal_paths(current_columns, di_edges + bi_edges,  # 得到所有因果路径
                                obj_columns)

    # delete copy column
    result_list = {}
    for obj_column in obj_columns:
        obj_paths = paths[obj_column]
        for sublist in obj_paths:
            last_elements = sublist[-1]
            if last_elements not in result_list:
                result_list[last_elements] = sublist
        paths[obj_column] = list(result_list.values())

    # compute causal paths
    if len(obj_columns) < 2:
        # single objective
        for key, val in paths.items():
            if len(paths[key]) > NUM_PATHS:
                paths = CM.compute_path_causal_effect(samples_df, paths[key], G,  # 根据平均因果效应选择前k个因果路径
                                                  NUM_PATHS)
                # paths = paths[obj_columns[0]]
            else:
                paths = paths[obj_columns[0]]

        # compute individual treatment effect in a path

        # recommond_config = CM.compute_individual_treatment_effect(samples_df, paths,  # 个体因果效应估计并选择配置
        #                                                 query, obj_columns, ref[obj_columns[0]],
        #                                                 previous_config, set_values_range, var_types)
        # sampled_config = recommond_config.to_dict()

    # extract tunning columns
    tune_conf = {}
    for path in paths:
        key = path[-1]
        if key in app_setting.keys():
            tune_conf[key] = app_setting[key]
    print("Tuning Partmenter:", tune_conf.keys())
    with open('/home/hmj/tuning_spark/target/target_spark/results/causal_graph/causal_path', 'a+') as f:
        f.writelines(str(tune_conf.keys()) + os.linesep)

    # bo iteration
    bo_iteration = 30
    for bo_id in range(bo_iteration):
        if bo_id == 0:
            # initation optimizer
            optimizer = create_optimizer(test_config.optimizer.name, configs=tune_conf,
                                         extra_vars=test_config.optimizer.extra_vars)
            if hasattr(optimizer, 'set_status_file'):
                optimizer.set_status_file(result_dir / 'optimizer_status')
            # add initial sample
            default_conf = get_default(app_setting) #previous_config.to_dict()  # like a initial_conf
            sampled_config_numeric = get_default_narrow(tune_conf)   # , sampled_config , previous_config.to_dict().copy()
            optimizer.add_observation((sampled_config_numeric, samples_df.iloc[0, -1]))
            print("Default config execute opty:",samples_df.iloc[0, -1])
            # optimizer.add_observation((sampled_config_numeric, ref[obj_columns[0]]))
            print("Initatial Optimizer!")
            print("========================================================")

        # if bo_id ==0:
        #     sampled_config = default_conf.copy()
        # else:
        #  recommond configs
        sampled_config_numeric, sampled_config = optimizer.get_conf(default_conf.copy())

        # translate configs
        trans_config = sampled_config.copy()
        trans_config = translate_configs(app_setting, trans_config, 0)  # 0: only trans num type
        sampled_config = translate_configs(app_setting, sampled_config, 1)  # 1: tans num type and vaule
        # if bo_id == 0:
        #     default_conf = trans_config

        #test      # setting config and default vaule
        confs = save_tune_conf(task_id*bo_iteration+bo_id, tune_columns, sampled_config)  # print the tuning parameters value
        save_confs = save_tune_conf(task_id*bo_iteration+bo_id, tune_columns, trans_config)  # save the tuning parameters value after transform
        sampled_os_config, sampled_app_config = divide_config(sampled_config, os_setting=os_setting, app_setting=app_setting)
        # if tune_app is off, just give sample_app_config a default app_setting value
        if test_config.tune_app is False:
            sampled_app_config = get_default(app_setting)
            sampled_app_config = translate_configs(app_setting, sampled_app_config, 1)

        # - dump configs
        os_config_path = result_dir / f'{task_id*bo_iteration+bo_id}_os_config.yml'
        os_config_path.write_text(yaml.dump(sampled_os_config, default_flow_style=False))
        app_config_path = result_dir / f'{task_id*bo_iteration+bo_id}_app_config.yml'
        app_config_path.write_text(yaml.dump(sampled_app_config, default_flow_style=False))

        start = time.time()
        result_list = []
        skip= False
        # print(default_conf)
        for rep in range(test_config.optimizer.repitition):
            await single_test(                 # add collection performance parameters vaule when runing
                task_name=test_config.task_name,task_id=task_id*bo_iteration+bo_id,rep=rep,tester=tester,testee=testee,slave1=slave1,slave2=slave2,
                tune_os=(task_id != 0 and test_config.tune_os),clients=test_config.clients,_skip=skip
            )

            _print(f'{task_id*15+bo_id} - {rep}: parsing result...')
            result = parse_result(
                tester_name=test_config.tester,result_dir=result_dir,task_id=task_id*bo_iteration+bo_id,rep=rep,printer=_print
            )

            result_list.append(- result)
            _print(f'{task_id*bo_iteration+bo_id} - {rep}: done.')

            #hmj # collection one samples
            apperf_filename = '/home/hmj/tuning_spark/target/target_spark/results/temp_feature_vector'
            hdperf_filename = '/home/hmj/cur'
            apperf_one_df = get_log_events_samples(apperf_filename, apperf_columns)  #
            apperf_df = pd.concat([apperf_df, apperf_one_df], axis=0)
            hdperf_one_df = get_one_sample(hdperf_filename, hdperf_columns)  #
            hdperf_df = pd.concat([hdperf_df, hdperf_one_df], axis=0)
            #

        # add observation sample        # add sample / updata casual graph
        metric_result = mean(result_list) if len(result_list) > 0 else .0
        optimizer.add_observation((sampled_config_numeric, metric_result))
        if hasattr(optimizer, 'dump_state'):
            optimizer.dump_state(result_dir / f'{task_id*bo_iteration+bo_id}_optimizer_state')

        #hmj
        # caculate perf_columns means values of one_configs
        apperf_df = pd.DataFrame(apperf_df.mean()).transpose()
        hdperf_df = pd.DataFrame(hdperf_df.mean()).transpose()
        # save origination data
        save_configs = save_confs
        save_configs.extend(apperf_df.iloc[0].values.tolist())
        save_configs.extend(hdperf_df.iloc[0].values.tolist())
        save_configs.append(- metric_result)
        save_configs = pd.DataFrame([save_configs])
        save_configs.columns = save_columns # current_columns  #
        save_configs = save_configs.round(3)
        iteration_configs = save_configs[current_columns]
        samples_df = pd.concat([samples_df, iteration_configs], axis=0, ignore_index=True)   # build causal samples updata
        save_configs.to_csv('/home/hmj/tuning_spark/target/target_spark/data/running/micro/wordcount_causal_bo_samples.csv', mode='a', header=False, index=False)
        # clean temp data
        apperf_df = pd.DataFrame(columns=apperf_columns)
        hdperf_df = pd.DataFrame(columns=hdperf_columns)
        ##

        end = time.time() - start
        logger.info('the time of evaluate configs {} is {}'.format(int(task_id*bo_iteration+bo_id), end))
        logger.info('the result of task_id {} is {}'.format(int(task_id*bo_iteration+bo_id), metric_result))
        logger.info('the config of task_id {} is {}'.format(int(task_id*bo_iteration+bo_id), confs))

    if task_id == causal_id-1:
        continue
    graph_start = time.time()
    ## fci: build
    G, di_edges, bi_edges = run_fci_loop(CM, samples_df, tabu_edges,
                                         current_columns, obj_columns, NUM_PATHS)
    ##
    # ges: build
    # G, di_edges, bi_edges = run_ges_loop(samples_df, tabu_edges,
    #                                      current_columns, obj_columns)
    ##
    # mmhc: build
    # G, di_edges, bi_edges = run_mmhc_loop(samples_df, tabu_edges,
    #                                       current_columns, obj_columns)
    ##
    graph_end = time.time() - graph_start
    with open('/home/hmj/tuning_spark/target/target_spark/results/causal_graph/causal_path', 'a+') as f:
        f.writelines(str(di_edges) + str(bi_edges) + os.linesep)
    # draw graph
    draw_graph(di_edges + bi_edges, task_id+1)
    logger.info('the time of build graph {} is {}'.format(task_id+1, graph_end))
    ##

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
