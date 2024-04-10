from ..bayes_opt import BayesianOptimization
from ..bayes_opt.helpers import acq_max, UtilityFunction
from random import sample, randint, random,choice,uniform

from ..other import random_sample


def noop(*kargs, **kwargs):
  # stub function for bo
  return None

class BOblissOptimizer():#define of bo,and set the value of space /conf
  def __init__(self,space,conf={}):
    conf = {
        **conf,
        'pbounds': space,
    }
    #print("try to see the value of conf")
    #print(conf)
    self.space = space
    self.conf = conf
#########conf contains acq, use; else use default
    self.acq = conf.get('acq', 'ucb') # get the value of acq function
    self.kappa = conf.get('kappa', 2.576)
    self.xi = conf.get('xi', 0.0)
    try:
      del conf['acq'], conf['kappa'], conf['xi']
    except:
      pass
    self.bo = BayesianOptimization(**self._make_config(conf))
    # return the conf with the type of dict and add the option of f
  def _make_config(self, conf):
    return {
        **conf,
        'f': noop
    }

  def add_observation(self,ob):
    # ob: (x, y) while x is argument dict

    _x,y=ob
    # dict to tuple regarding keys in self.space
    x = []
    if isinstance(_x,dict):
      for k in self.space.keys():
        x.append(_x[k])
    else:
      for k in _x:
        x.append(k)
# add ob into bo space
    try:
      #space.add_observation(x, y) is define in the TargetSpace.py file
      self.bo.space.add_observation(x, y)
    except KeyError as e:
      # get exception message
      msg, = e.args
      raise Exception(msg)
    self.bo.gp.fit(self.bo.space.X, self.bo.space.Y)

  def get_conf(self):
    acq = self.acq
    kappa = self.kappa
    xi = self.xi
    # bo recalculates next best conf
    # codes below are adapted from implementation of bo.maximize

    # assert self.bo.space.Y is not None and len(
    #     self.bo.space.Y) > 0, 'at least one observation is required before asking for next configuration'
    #
    if self.bo.space.Y is None or len(self.bo.space.Y) == 0:
      #print(self.bo.space.Y)
      x_max = self.bo.space.random_points(1)[0]
      y_predict=None
    else:
      #get the preudo-point_x,and add it to the set of observation
      x_max,y_predict= acq_max(
        ac=UtilityFunction(
          kind=acq,
          kappa=kappa,
          xi=xi,
        ).utility,
        gp=self.bo.gp,
        y_max=self.bo.space.Y.max(),
        bounds=self.bo.space.bounds,
        random_state=self.bo.random_state,
        **self.bo._acqkw
      )
    # check if x_max repeats
    if x_max in self.bo.space:
      x_max = self.bo.space.random_points(1)[0]
      y_predict=None
    return self._convert_to_dict(x_max),y_predict

  def _convert_to_dict(self, x_array):
    return dict(zip(self.space, x_array))

class BOblissBayesianOptimizer(BOblissOptimizer):
  # Processing parameter space: Continuous and discrete
  def __init__(self, config, bo_conf={}):
    self._config = {**config}
    #print(self._config)
    #bo_space = {}
    # for k, v in self._config.items():
    #   v_range = v.get('range')
    #   if v_range:  # discrete ranged parameter
    #     bo_space[k] = (0, len(v_range))  # note: right-close range
    #   else:
    #     bo_space[k] = (v['min'], v['max'])
    reduce_space={}
    for k,v in self._config.items():
      reduce_space[k]=(-1,1)
    super().__init__(reduce_space, bo_conf)
  # get conf and convert to legal config
  def get_conf(self,current_ob_x,current_ob_y):
    x_view = self.bo.space.X
    y_view = self.bo.space.Y
    length = self.bo.space._length
    if length >= 5:
      number = int(length)
      choose_number = sample(range(0, length), number-1)
      print(choose_number)
      pseudo_point_x = []
      pseudo_point_y = []
      for i in choose_number:
        pseudo_point_x.append(x_view[i])
        pseudo_point_y.append(y_view[i])
      for x, y in zip(pseudo_point_x, pseudo_point_y):
         for i in range(len(x)):
            x[i] = uniform(x[i] - 0.05, x[i] + 0.05)
         self.add_observation((x, y))
    samples,y_predicts= super().get_conf()
    self.bo.space._length = length
    self.bo.space._Xarr = x_view
    self.bo.space._Yarr = y_view
    self.bo.space._Xview = x_view
    self.bo.space._Yview = y_view
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
    index_max = current_ob_y.index(max(current_ob_y))
    current_best_configD_for_copy = current_ob_x[index_max]
    current_best_configD = current_best_configD_for_copy.copy()
    for key in new_sample:
      current_best_configD[key]=int(new_sample[key])
    # first is continuous value, second is translated
    return samples, current_best_configD,y_predicts
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
