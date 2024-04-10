from ..bayes_opt import BayesianOptimization
from ..bayes_opt.helpers import acq_max, UtilityFunction
from random import sample, randint, random,choice,uniform

from ..other import random_sample


def noop(*kargs, **kwargs):
  return None


class BOppOptimizer():#define of bo,and set the value of space /conf
  def __init__(self,space,conf={}):
    conf = { **conf,'pbounds': space,}
    self.space = space
    self.conf = conf
    self.acq = conf.get('acq', 'ucb')
    self.kappa = conf.get('kappa', 2.576)
    self.xi = conf.get('xi', 0.0)
    try:
      del conf['acq'], conf['kappa'], conf['xi']
    except:
      pass
    # pass the value of bo,and return a bo objectm,and set the value of bo's attributes
    self.bo = BayesianOptimization(**self._make_config(conf))
    # return the conf with the type of dict and add the option of f
  def _make_config(self, conf):
    return {**conf,'f': noop}
  def add_observation(self,ob):
    _x,y=ob
    # dict to tuple regarding keys in self.space
    x = []
    if isinstance(_x,dict):
      for k in self.space.keys():
        x.append(_x[k])
    else:
      for k in _x:
        x.append(k)
    try:
      self.bo.space.add_observation(x, y)
    except KeyError as e:
      msg, = e.args
      raise Exception(msg)
    self.bo.gp.fit(self.bo.space.X, self.bo.space.Y)
  def get_conf(self):
    acq = self.acq
    kappa = self.kappa
    xi = self.xi
    # assert self.bo.space.Y is not None and len(
    #     self.bo.space.Y) > 0, 'at least one observation is required before asking for next configuration'
    if self.bo.space.Y is None or len(self.bo.space.Y) < 1:
      x_max = self.bo.space.random_points(1)[0]
    else:
      x_max ,y_predict= acq_max(
        ac=UtilityFunction(
          kind=acq,
          kappa=kappa,
          xi=xi
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
    return self._convert_to_dict(x_max)
  def _convert_to_dict(self, x_array):
    return dict(zip(self.space, x_array))

class BOhalfppBayesianOptimizer(BOppOptimizer):
  # Processing parameter space: Continuous and discrete
  def __init__(self, config, bo_conf={}):
    self._config = {**config}
    #print(self._config)
    bo_space = {}
    for k, v in self._config.items():
      v_range = v.get('range')
      if v_range:  # discrete ranged parameter
        bo_space[k] = (0, len(v_range))  # note: right-close range
      else:
        bo_space[k] = (v['min'], v['max'])
    reduce_space={}
    for k,v in self._config.items():
      reduce_space[k]=(-1,1)
    super().__init__(reduce_space, bo_conf) #super() is try to use the mathod of father class
  # get conf and convert to legal config
  def get_conf(self,default_conf):
    x_view = self.bo.space.X
    y_view = self.bo.space.Y
    length = self.bo.space._length
    if length >= 2:
      number = int(length/2)
      choose_number = sample(range(0, length), number)
      pseudo_point_x = []
      pseudo_point_y = []
      for i in choose_number:
        pseudo_point_x.append(x_view[i])
        pseudo_point_y.append(y_view[i])
      for x, y in zip(pseudo_point_x, pseudo_point_y):
         for i in range(len(x)):
            x[i] = uniform(x[i] - 0.05, x[i] + 0.05)
         self.add_observation((x, y))
    samples = super().get_conf()
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
    new_conf = default_conf
    for key in new_sample:
      new_conf[key] = int(new_sample[key])
    # first is continuous value, second is translated
    return samples, new_conf
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
