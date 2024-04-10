#!/usr/bin/python3.9
import asyncio
import yaml         # pyYaml == 5.1 ; numpy == 1.23.5
import sys
import logging
import random
from pathlib import Path
from statistics import mean
from tqdm import tqdm
from lib import parse_cmd,run_playbook,get_default,save_tune_conf,find_exist_task_result,\
    divide_config,_print,parse_result,get_default_narrow
from lib.optimizer import create_optimizer
from src.causal_mode import *
from src.deel_data import *

async def main(test_config,os_setting,app_setting,tune_conf):
  global feature_vector_path
  global runtime_path
  assert test_config.tune_os or test_config.tune_app, 'at least one of tune_app and tune_os should be True'

  # parameters     # initialize and add nodes for causal graph
  tune_configs=[]
  for key in tune_conf:
      tune_configs.append(key)
  os_setting = os_setting['options_vaules']
  app_setting = app_setting['options_vaules']

  # initation optimizer    # initialize causal model object
  optimizer = create_optimizer(test_config.optimizer.name,configs = tune_conf,extra_vars=test_config.optimizer.extra_vars)
  default_conf = 0

  if hasattr(optimizer, 'set_status_file'):
    optimizer.set_status_file(result_dir / 'optimizer_status')

  tester, testee, slave1,slave2= test_config.hosts.master,test_config.hosts.master,test_config.hosts.slave1,test_config.hosts.slave2

  # save config and performance
  logging.basicConfig(level=logging.INFO,format='[%(asctime)s][%(name)s][%(levelname)s][%(message)s]')
  logger=logging.getLogger('run-bo')
  handler=logging.FileHandler('run_information/run-bo.txt')
  handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(message)s]'))
  logger.addHandler(handler)

  for task_id in tqdm(range(test_config.optimizer.iter_limit)):

    # get recommand config of optimizer       # identify and compute causal paths （get config）
    if task_id == 0:
      sampled_config_numeric, sampled_config = get_default_narrow(tune_conf), get_default(app_setting)
      default_conf = get_default(app_setting)
      # optimizer.add_observation((sampled_config_numeric, 52))
      # sampled_config_numeric, sampled_config = optimizer.get_conf(default_conf)
    else:
      try:
          sampled_config_numeric, sampled_config = optimizer.get_conf(default_conf.copy())
      except StopIteration:
        return

    sampled_config = translate_configs(app_setting, sampled_config, 1)  # 1: tans num type and vaule
    #test      # setting config and default vaule
    confs=save_tune_conf(task_id, tune_conf,sampled_config)  # print the tuning parameters value
    # all_config = save_tune_conf(task_id, sampled_config,sampled_config)
    sampled_os_config, sampled_app_config = divide_config(sampled_config,os_setting=os_setting,app_setting=app_setting)
    # if tune_app is off, just give sample_app_config a default app_setting value
    if test_config.tune_app is False:
      sampled_app_config = get_default(app_setting)

    # - dump configs
    os_config_path = result_dir / f'{task_id}_os_config.yml'
    os_config_path.write_text(yaml.dump(sampled_os_config, default_flow_style=False))
    app_config_path = result_dir / f'{task_id}_app_config.yml'
    app_config_path.write_text(yaml.dump(sampled_app_config, default_flow_style=False))

    result_list = []
    skip= False
    for rep in range(test_config.optimizer.repitition):
        await single_test(                 # add collection performance parameters vaule when runing
            task_name=test_config.task_name,task_id=task_id,rep=rep,tester=tester,testee=testee,slave1=slave1,slave2=slave2,
            tune_os=(task_id != 0 and test_config.tune_os),clients=test_config.clients,_skip=skip
        )

        _print(f'{task_id} - {rep}: parsing result...')
        result = parse_result(
            tester_name=test_config.tester,result_dir=result_dir,task_id=task_id,rep=rep,printer=_print
        )
        result_list.append(- result)
        _print(f'{task_id} - {rep}: done.')

        # record and caculate means of

    # # get the default config with tune_config
    # if task_id == 0:
    #   default_conf = sampled_config

    # add observation sample        # add sample / updata casual graph
    metric_result = mean(result_list) if len(result_list) > 0 else .0
    optimizer.add_observation((sampled_config_numeric, metric_result))

    if hasattr(optimizer, 'dump_state'):
        optimizer.dump_state(result_dir / f'{task_id}_optimizer_state')

    logger.info('the result of task_id {} is {}'.format(task_id, metric_result))
    logger.info('the config of task_id {} is {}'.format(task_id, confs))

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
os_setting_path = proj_root / \
    f'target/{test_config.target}/new_os_configs_info.yml'
app_setting_path = proj_root / \
    f'target/{test_config.target}/new_app_configs_info.yml'
tune_conf_path = proj_root / \
    f'target/{test_config.target}/low_app_configs_A.yml'

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
os_setting = yaml.load(os_setting_path.read_text(), Loader=yaml.FullLoader)  # pylint: disable=E1101
app_setting = yaml.load(app_setting_path.read_text(), Loader=yaml.FullLoader)  # pylint: disable=E1101
tune_conf = yaml.load(tune_conf_path.read_text(), Loader=yaml.FullLoader)

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
