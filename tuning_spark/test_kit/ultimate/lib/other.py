# -*- coding:utf-8 -*-

import asyncio
import json
import sys
import yaml
import re
import traceback
import os
import numpy as np
import itertools
from datetime import datetime
from math import inf
from pathlib import Path
from random import randint, random, choice
from statistics import mean, stdev

async def run_playbook(playbook_path, tags='all', **extra_vars):
  vars_json_str = json.dumps(extra_vars)
  command = f'ansible-playbook {playbook_path} --extra-vars=\'{vars_json_str}\' --tags={tags}'
  process = await asyncio.create_subprocess_shell(
      cmd=command,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
      stdin=asyncio.subprocess.PIPE,
  )
  print(command)
  stdout, stderr = await process.communicate()
  if process.returncode != 0:
    raise RuntimeError(
        f'Error running playbook. Below is stdout:\n {stdout.decode("utf-8")}\nand stderr: {stderr.decode("utf-8")}\n')
  return stdout.decode('utf-8'), stderr.decode('utf-8')

def parse_cmd(run_info):
  args = sys.argv
  try:
    # MEIassert len(args) > 1, 'too few arguments'
    # conf_path = ../../target/hbase/tests/bo-ei.yml
    # *others = task_name=hbase-bo-test exist=delete
    others = ['task_name=spark-bopp-test', 'exist=delete']
    conf = yaml.load(
        Path("../../target/target_spark/tests/"+f'{run_info}').resolve().read_text(), Loader=yaml.FullLoader  # pylint: disable=E1101
    )
    regexp = re.compile(r'(.+)=(.+)')
    for other in others:
      match = regexp.match(other.strip())
      k, v = match.groups()
      assign(conf, k, v)
    return TestConfig(conf)
  except Exception as e:
    print(e.args)
    print(traceback.print_stack(e))
    print('Usage: python run.py <path_to_conf> [path.to.attr=value]')

class TestConfig:
  def __init__(self, obj):
    self.task_name = obj['task_name']
    self.target = obj['target']
    self.hosts = HostConifg(obj['hosts'])
    self.workload = obj['workload']
    self.tune_os = obj['tune_os']
    self.tune_app = obj['tune_app']
    self.exist = obj.get('exist')
    self.optimizer = OptimizerConfig(obj['optimizer'])
    self.clients = obj['clients']
    self.tester = obj['tester']

class HostConifg:
  def __init__(self, obj):
    self.master = obj['master']
    self.slave1 = obj['slave1']
    self.slave2 = obj['slave2']

class OptimizerConfig:
  def __init__(self, obj):
    self.name = obj['name']
    self.iter_limit = obj['iter_limit']
    self.reboot_interval = obj['reboot_interval']
    self.repitition = obj['repitition']
    self.extra_vars = obj.get('extra_vars', {})

    if self.iter_limit < 0:
      self.iter_limit = inf

def assign(obj, path, value):
  keys = path.split('.')
  for k in keys[:-1]:
    v = obj.get(k)
    if v is None:
      obj[k] = {}
    elif type(v) is not dict:
      raise Exception(f'error while assigning {path} with {value} on {obj}.')
    obj = obj[k]
  try:
    value = int(value)
  except:
    pass
  if str(value) in ('yes', 'true'):
    value = True
  if str(value) in ('no', 'false'):
    value = False
  obj[keys[-1]] = value

def random_sample(config):
  res = {}
  for k, conf in config.items():
    numer_range = conf.get('range')
    if numer_range is None:
      minn = conf.get('min')
      maxx = conf.get('max')
      allow_float = conf.get('float', False)
      res[k] = random() * (maxx - minn) + minn \
          if allow_float else randint(minn, maxx)
    else:
      res[k] = choice(numer_range)
    if type(res[k]) is bool:
      res[k] = str(res[k]).lower()
  return res

def get_default(config):
  res = {}
  for k, conf in config.items():
    res[k] = conf['default']
    if type(res[k]) is bool:
      res[k] = str(res[k]).lower()
  return res

def get_default_narrow(config):
  res = []
  for k,conf in config.items():
    c_range = conf.get('range')
    if c_range:
      if type(conf['default']) == type('str') :
        res.append(-1.11)
      else:
        res.append(-1 + ((conf['default'] - c_range[0]) / (c_range[len(c_range) - 1] - c_range[0])) * 2)
    else:
      res.append(-1 + ((conf['default'] - conf['min']) / (conf['max'] - conf['min'])) * 2)
  return res

def choose_metric_results(metric_results_list):
   #  print("old: ", metric_results_list)
   #  metric_results_list.remove(min(metric_results_list))
   #  metric_results_list.remove(max(metric_results_list))
   #  print("new: ", metric_results_list)
     result = []
     u = mean(metric_results_list)
     std = stdev(metric_results_list)
     print("Mean Value = ", u, "Standard Deviation =", std)

     for i in range(len(metric_results_list)):
       if abs(metric_results_list[i] - u) <= std:
          result.append(metric_results_list[i])
     return metric_results_list

def save_tune_conf(task_id,tune_conf,sampled_config):
  curr_time = datetime.now()
  confs = []
  time = curr_time.strftime("%m-%d")
  path = '/home/hmj/tuning_spark/target/target_spark/results/tune_config'
  if task_id == 0:
    isExists = os.path.exists(path)
    if not isExists:
      os.makedirs(path)
    with open(path + '/' + time, 'a+') as f:
      for key in tune_conf:
        f.writelines('{:<30}'.format(key))
  with open(path + '/' + time, 'a+') as f:
    f.writelines(os.linesep)
    f.writelines('{:<10}'.format(task_id))
    for key in tune_conf:
      f.writelines('{:<30}'.format(sampled_config[key]))
      confs.append(sampled_config[key])
    f.writelines(os.linesep)
    return confs

def find_exist_task_result(result_dir): # look for the file under the results
  task_id = -1
  regexp = re.compile(r'(\d+)_.+')
  if result_dir.exists():
    for p in result_dir.iterdir():
      if p.is_file():
        res = regexp.match(p.name)
        if res:
          task_id = max(task_id, int(res.group(1)))
  return None if task_id == -1 else task_id

def divide_config(sampled_config, os_setting, app_setting):# divide the sampled_config into os and app
  for k in sampled_config.keys():
    if type(sampled_config[k]) is bool:
      # make sure no uppercase 'True/False' literal in result
      sampled_config[k] = str(sampled_config[k]).lower()
    elif type(sampled_config[k]) is np.float64:
      sampled_config[k] = float(sampled_config[k])
  sampled_os_config = dict(
      ((k, v) for k, v in sampled_config.items() if k in os_setting)
  )
  sampled_app_config = dict(
      ((k, v) for k, v in sampled_config.items() if k in app_setting)
  )
  return sampled_os_config, sampled_app_config

def current_feature(feature_vector_path):
  with open(feature_vector_path, encoding='utf-8') as f:
    feature = []
    for line in f.readlines():
      line = line.split()
      stage_feature = line[0]
      for i in range(1, len(line)):
        if (i-1)%5 == 0:
          continue
        feature.append(float(line[i]))
      break
    np_feature = np.array(feature, ndmin=2)
    return np_feature, stage_feature
def _print(msg):
  print(f'[{datetime.now()}] - {msg}')