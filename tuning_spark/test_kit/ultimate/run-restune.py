import asyncio

import pandas as pd
import yaml
import sys
import logging
import random
from pathlib import Path
from statistics import mean
from tqdm import tqdm
from lib import parse_cmd,run_playbook,get_default,save_tune_conf,find_exist_task_result,\
    divide_config,current_feature,parse_result,_print,get_default_narrow
from lib.optimizer import create_optimizer
from lib.decision_tree import read_history_data,ModelPool
from lib.bayes_opt import acq_max2,build_configuration
from src.deel_data import *

async def main(test_config,os_setting,app_setting,tune_conf):
  global feature_vector_path
  global runtime_path
  assert test_config.tune_os or test_config.tune_app, 'at least one of tune_app and tune_os should be True'

  current_ob_x = []
  current_ob_y = []
  tune_configs=tune_conf.copy()
  recom_alltime = 0
  # for key in tune_conf:
  #     tune_configs.append(key)

  # columns list
  obj_columns = ['run_times']
  os_columns = os_setting['os_columns']
  app_columns = app_setting['app_columns']
  hdperf_columns = os_setting['hard_perf_columns']
  apperf_columns = os_setting['app_perf_columns']
  # select columns
  os_columns = []
  # use columns
  perf_columns = apperf_columns + hdperf_columns
  tune_columns = app_columns + os_columns
  current_columns = tune_columns + perf_columns + obj_columns
  # get tuning columns values list
  app_setting = get_option_values(app_columns, app_setting['options_vaules'])
  os_setting = get_option_values(os_columns, os_setting['options_vaules'])
  tune_para_setting = {**app_setting, **os_setting}
  tune_column =  ['jvm_par_gc_thread', 'jvm_new_ratio', 'spark_locality_wait', 'jvm_max_gc_pause', 'spark_default_parallelism', 'spark_memory_fraction','spark_kryoserializer_buffer', 'jvm_survivor_ratio', 'spark_executor_memory', 'spark_broadcast_blockSize']

  # tune_column = tune_columns.copy()
  tune_conf = {}
  for key in tune_column:
      if key in tune_configs.keys():
          tune_conf[key] = tune_configs[key]
  ##

  optimizer = create_optimizer(test_config.optimizer.name,configs = tune_conf,extra_vars=test_config.optimizer.extra_vars)
  default_conf = None
  default_time = 0
  if hasattr(optimizer, 'set_status_file'):
    optimizer.set_status_file(result_dir / 'optimizer_status')
  x, y, z,s = read_history_data()   # history data
  for key,_ in x.items():
      temp_df = pd.DataFrame(x[key],columns=tune_columns)
      temp_df = temp_df[tune_column]
      x[key] = temp_df.values
  modelpool = ModelPool(x, y, z, s).BuildBOModels(test_config.optimizer.name,tune_conf,test_config.optimizer.extra_vars)

  tester, testee, slave1,slave2= test_config.hosts.master,test_config.hosts.master,test_config.hosts.slave1,test_config.hosts.slave2
  logging.basicConfig(level=logging.INFO,format='[%(asctime)s][%(name)s][%(levelname)s][%(message)s]')
  logger=logging.getLogger('run-restune')
  handler=logging.FileHandler('run_information/restune.txt')
  handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(message)s]'))
  logger.addHandler(handler)

  x_tries = build_configuration() # ?
  for task_id in tqdm(range(test_config.optimizer.iter_limit)):
    start = time.time()
    if task_id == 1:
        np_feature, stage_feature = current_feature(feature_vector_path)
        for i in range(len(modelpool)):
            modelpool[i].getSimilarity(np_feature, stage_feature)

    if task_id == 0:
      sampled_config_numeric, sampled_config = get_default_narrow(tune_conf), get_default(app_setting)

    # elif task_id <= 5:
    #     try:
    #         sampled_config_numeric, sampled_config = optimizer.get_conf(default_conf)
    #     except StopIteration:
    #         return
    elif task_id > 0:
        try:
          sampled_config_numeric = acq_max2(task_id,x_tries,modelpool,optimizer,current_ob_x,current_ob_y,default_conf)
          sampled_config = optimizer.translate_conf(default_conf.copy(),sampled_config_numeric)
        except StopIteration:
          return

    C_start = time.time()
    trans_config = sampled_config.copy()
    trans_config = translate_configs(app_setting, trans_config, 0)  # 0: only trans num type
    if task_id == 0:
        default_conf = trans_config.copy()
    sampled_config = translate_configs(app_setting, sampled_config, 1)  # 1: tans num type and vaule
    confs=save_tune_conf(task_id, tune_conf, sampled_config)
    save_confs = save_tune_conf(task_id, app_setting,trans_config)  # save the tuning parameters value after transform
    sampled_os_config, sampled_app_config = divide_config(sampled_config,os_setting=os_setting,app_setting=app_setting)
    if test_config.tune_app is False:
      sampled_app_config = get_default(app_setting)
    # - dump configs
    os_config_path = result_dir / f'{task_id}_os_config.yml'
    os_config_path.write_text(yaml.dump(sampled_os_config, default_flow_style=False))
    app_config_path = result_dir / f'{task_id}_app_config.yml'
    app_config_path.write_text(yaml.dump(sampled_app_config, default_flow_style=False))
    result_list = []
    skip= False
    if task_id == 0:
        iteration_num = test_config.optimizer.repitition
    else:
        iteration_num = 1
    for rep in range(iteration_num):
        await single_test(
            task_name=test_config.task_name, task_id=task_id, rep=rep, tester=tester, testee=testee, slave1=slave1,slave2=slave2,
            tune_os=(task_id != 0 and test_config.tune_os), clients=test_config.clients, _skip=skip
        )
        _print(f'{task_id} - {rep}: parsing result...')
        result = parse_result(
            tester_name=test_config.tester, result_dir=result_dir, task_id=task_id, rep=rep, printer=_print
        )
        result_list.append(- result)
        _print(f'{task_id} - {rep}: done.')
    C_end = time.time() - C_start

    if task_id == 0:
        save_index = sorted(range(len(result_list)), key=lambda k: result_list[k], reverse=True)[1:-1]
        result_list = [result_list[i] for i in save_index]
    else:
        save_index = [0]
    metric_result = mean(result_list) if len(result_list) > 0 else .0
    if task_id == 0:
        default_time = abs(metric_result)
    optimizer.add_observation((sampled_config_numeric, (default_time-abs(metric_result))/default_time))
    current_ob_x.append(list(sampled_config_numeric))
    current_ob_y.append(abs(metric_result))
    end = time.time() - start
    recom_alltime = recom_alltime + (end - C_end)
    #hmj
    # save origination data
    save_configs = save_confs
    save_configs.append(- metric_result)
    save_configs = pd.DataFrame([save_configs])
    save_configs.columns = tune_columns + obj_columns
    save_configs = save_configs.round(3)
    save_configs.to_csv('/home/hmj/tuning_spark/target/target_spark/data/running/ml/bayes_restune_samples.csv', mode= 'a', header= False, index = False)
    if hasattr(optimizer, 'dump_state'):
        optimizer.dump_state(result_dir / f'{task_id}_optimizer_state')
    logger.info('the result of task_id {} is {}'.format(task_id, metric_result))
    logger.info('the config of task_id {} is {}'.format(task_id, confs))
  logger.info('All Recommend Time:{}'.format(round(recom_alltime,2)))


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
    if task_id == 0 and rep == 0:
  #    - deploy db
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

      _print(f'{task_id} - {rep}: hadoop_master first deploying...')
      stdout_hadoop, stderr_hadoop = await run_playbook(
        deploy_hadoop_playbook_path,
        host=tester,
        task_name=task_name,
        task_id=task_id,
        task_rep=rep,
      )
      _print(f'{task_id} - {rep}: hadoop_master first done.')
  # #
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
    await run_playbook(
        tester_playbook_path,
        host=testee,
        target=tester,
        task_name=task_name,
        task_id=task_id,
        task_rep=rep,
        workload_path=str(workload_path),
        n_client=clients
    )
    _print(f'{task_id} - {rep}: hibench done.')

    _print(f'{task_id} - {rep}: clean logs...')
    await run_playbook(
        clean_playbook_path,
        host=testee,
        target=tester,
        task_name=task_name,
        task_id=task_id,
        task_rep=rep,
        workload_path=str(workload_path),
        n_client=clients
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
run_info='restune.yml' # !!!!!跑restune的时候记得改这个文件
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
# os_setting = yaml.load(os_setting_path.read_text())  # pylint: disable=E1101
# app_setting = yaml.load(app_setting_path.read_text())  # pylint: disable=E1101
# tune_conf = yaml.load(tune_conf_path.read_text())

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
