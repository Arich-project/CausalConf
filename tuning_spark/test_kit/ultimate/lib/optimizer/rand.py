from random import  random
from ..other import random_sample,get_default
import numpy as np
def noop(*kargs, **kwargs):
  return None

class RandOptimizer():
  # Processing parameter space: Continuous and discrete
  def __init__(self, config):
    self._config = {**config}
    bo_space = {}
    for k, v in self._config.items():
      v_range = v.get('range')
      if v_range:
        bo_space[k] = (0, len(v_range))
      else:
        bo_space[k] = (v['min'], v['max'])
    self.reduce_space={}
    for k,v in self._config.items():
      self.reduce_space[k]=(-1,1)
  def get_conf(self,app_setting):
    samples={}
    for key in self.reduce_space:
      samples[key]=random()*(1-(-1))+(-1)
    res=get_default(app_setting)
    #change the scala of sample into normally
    bo_space={}
    for k,v in self._config.items():
      v_range=v.get('range')
      if v_range:
        bo_space[k]=(0,len(v_range))
      else:
        bo_space[k]=(v['min'],v['max'])
    new_sample={}
    for key in samples:
      new_sample[key]=self._rescale(samples[key], bo_space[key])
    for key in new_sample:
      res[key]=int(new_sample[key])
    # first is continuous value, second is translated
    return samples, res
  #chage the scala
  def _rescale(self, origin_v, to_scale, origin_scale=(-1, 1)):
    a, b = origin_scale
    c, d = to_scale
    if origin_v > b:
       origin_v = b
    if origin_v < a:
       origin_v = a
    to_v=origin_v
    to_v *= (d - c) / (b - a)  # scale
    to_v += c - a * (d - c) / (b - a)  # offset
    return to_v
  def random_sample(self):
    result = {}
    for k, v in self._config.items():
      v_range = v.get('range')
      if v_range:
        result[k] = random() * len(v_range)
      else:
        minn, maxx = v.get('min'), v.get('max')
        result[k] = random() * (maxx - minn) + minn
    return result, self._translate(result)

  def _translate(self, sample):
    result = {}
    # orders in sample are the same as in _config dict
    #   see: https://github.com/fmfn/BayesianOptimization/blob/d531dcab1d73729528afbffd9a9c47c067de5880/bayes_opt/target_space.py#L49
    #   self.bounds = np.array(list(pbounds.values()), dtype=np.float)
    for sample_value, (k, v) in zip(sample.values(), self._config.items()):
      v_range = v.get('range')
      if v_range:
        try:
          index = int(sample_value)
          if index == len(v_range):
            index -= 1
          result[k] = v_range[index]
        except Exception as e:
          print('ERROR!')
          print(k, sample_value)
          print(v_range)
          raise e
      else:
        is_float = v.get('float', False)
        result[k] = sample_value if is_float else int(sample_value)
    #print(result)
    return result
