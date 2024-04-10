from __future__ import print_function
from __future__ import division
import numpy as np
import itertools
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import minimize
import random
import  heapq
def build_configuration(n_warmup=100000):
  bounds = [[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]]
  bounds = np.asarray(bounds)
  x_uniform = []
  for i in range(10):
    num = np.linspace(bounds[i,0], bounds[i,1], num=5, dtype=float)
    list = [num[1], num[2],num[3]]
    x_uniform.append(list)
  x_uniforms = []
  for item in itertools.product(*x_uniform):
    x_uniforms.append(item)
  x_uniforms = np.asarray(x_uniforms)
  n_warmup = n_warmup - x_uniforms.shape[0]
  x_random = np.random.uniform(bounds[:,0], bounds[:,1],size = (n_warmup, 10))
  x_tries = np.concatenate((x_uniforms, x_random))
  return x_tries

def acq_max2(task_id,x_tries,modelpool,optimizer,current_ob_x,current_ob_y,default_conf):
  if task_id <= 7:
    all_mean = 0
    all_std  = 0
    for i in range(len(modelpool)):
      mean, _= modelpool[i].model.bo.gp.predict(x_tries, return_std=True)
      mean = np.asarray(mean)
      all_mean += mean * modelpool[i].SimilarityWeight
    _,std = optimizer.bo.gp.predict(x_tries,return_std=True)
    std = np.asarray(std)
    ys = all_mean + 2.256*std
    x_max = np.clip(x_tries[ys.argmax()], -1, 1)
    if x_max in optimizer.bo.space:
      x_max = optimizer.bo.space.random_points(1)[0]
    return x_max
  else :
    bo_err,all_sort_err = CalculateSortError(modelpool,x_tries,optimizer,current_ob_x,current_ob_y)
    all_mean = 0
    for i in range(len(modelpool)):
      mean, _ = modelpool[i].model.bo.gp.predict(x_tries, return_std=True)
      mean = np.asarray(mean)
      all_mean += mean * ((modelpool[i].SortErrWeight)/all_sort_err)
    mean, std = optimizer.bo.gp.predict(x_tries,return_std=True)
    all_mean += mean*bo_err
    std = np.asarray(std)
    ys = all_mean + 2.576 * std
    x_max = np.clip(x_tries[ys.argmax()], -1, 1)
    if x_max in optimizer.bo.space:
      x_max = optimizer.bo.space.random_points(1)[0]
    return x_max

def CalculateSortError(modelpool,x_tries,optimizer,current_ob_x,current_ob_y):
  all_sort_err = 0
  for i in range(len(modelpool)+1):
    current_ob_x=np.asarray(current_ob_x)
    if i == len(modelpool):
      mean,std = optimizer.bo.gp.predict(x_tries,return_std=True)
    else:
      mean,std = modelpool[i].model.bo.gp.predict(current_ob_x,return_std=True)
    y_predict = mean + 2.256* std
    y_predict_value = dict(zip(range(len(y_predict)), y_predict))
    y_really_value = dict(zip(range(len(current_ob_y)), current_ob_y))
    result1_sort = list(np.asarray(sorted(y_predict_value.items(), key=lambda x: x[1], reverse=False))[:,0])
    result2_sort = list(np.asarray(sorted(y_really_value.items(), key=lambda x: x[1], reverse=False))[:,0])
    k = 0
    sort_err = 0
    while k < len(y_predict_value):
      j = k + 1
      while j < len(y_really_value):
        if ((result1_sort.index(k) - result1_sort.index(j)) * (result2_sort.index(k) - result2_sort.index(j))) < 0:
          sort_err += 1
        j += 1
      k += 1
    if i == len(modelpool):
      bo_err = sort_err
    else:
      modelpool[i].SortErrWeight = sort_err
    all_sort_err+=sort_err
  return  bo_err,all_sort_err

def acq_max(ac, gp, y_max, bounds, random_state,n_warmup=100000, n_iter=250):
  """
  A function to find the maximum of the acquisition function

  It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
  optimization method. First by sampling `n_warmup` (1e5) points at random,
  and then running L-BFGS-B from `n_iter` (250) random starting points.

  Parameters
  ----------
  :param ac:
      The acquisition function object that return its point-wise value.

  :param gp:
      A gaussian process fitted to the relevant data.

  :param y_max:
      The current maximum known value of the target function.

  :param bounds:
      The variables bounds to limit the search of the acq max.

  :param random_state:
      instance of np.RandomState random number generator

  :param n_warmup:
      number of times to randomly sample the aquisition function

  :param n_iter:
      number of times to run scipy.minimize

  Returns
  -------
  :return: x_max, The arg max of the acquisition function.
  """
  # Warm up with random points
  x_uniform=[]
  for i in range(bounds.shape[0]):
    num = np.linspace(bounds[i,0],bounds[i,1],num=5,dtype=float)
    lists=[num[1],num[2],num[3]]
    x_uniform.append(lists)
  x_uniforms=[]
  for item in itertools.product(*x_uniform):
    x_uniforms.append(item)
  x_uniforms=np.asarray(x_uniforms)
  n_warmup=n_warmup-x_uniforms.shape[0]
  x_random = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                 size=(n_warmup, bounds.shape[0]))
  x_random = np.asarray(x_random)
  x_tries =np.concatenate((x_uniforms,x_random))
  ys,std= ac(x_tries, gp=gp, y_max=y_max)
  # x_max=[]
  # x_max.append(xx_max)
  # x_max.append(xx_max)
  # max_acq = ys.max()
  # acs = UtilityFunction(
  #   kind='ucb',
  #   kappa=2.576,
  #   xi=0,
  # ).utility
  # yss=acs(x_max,gp=gp,y_max=y_max)
  # Explore the parameter space more throughly
  # x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
  #                                size=(n_iter, bounds.shape[0]))
  #try to see the result of delete the second find underhere
  # for x_try in x_seeds:
  #   # Find the minimum of minus the acquisition function
  #   #
  #   res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
  #                  x_try.reshape(1, -1),
  #                  bounds=bounds,
  #                  method="L-BFGS-B")
  #
  #   # See if success
  #   if not res.success:
  #     continue
  #
  #   # Store it if better than previous minimum(maximum).
  #   if max_acq is None or -res.fun[0] >= max_acq:
  #    # x_max = res.x
  #     x_max=x_try
  #     max_acq = -res.fun[0]
  return np.clip(x_tries[ys.argmax()], bounds[:, 0], bounds[:, 1]), ys[ys.argmax()]

def acq_max_bank(ac,bo,gp, y_max, bounds, random_state,n_warmup=100000, n_iter=250):
  """
  A function to find the maximum of the acquisition function

  It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
  optimization method. First by sampling `n_warmup` (1e5) points at random,
  and then running L-BFGS-B from `n_iter` (250) random starting points.

  Parameters
  ----------
  :param ac:
      The acquisition function object that return its point-wise value.

  :param gp:
      A gaussian process fitted to the relevant data.

  :param y_max:
      The current maximum known value of the target function.

  :param bounds:
      The variables bounds to limit the search of the acq max.

  :param random_state:
      instance of np.RandomState random number generator

  :param n_warmup:
      number of times to randomly sample the aquisition function

  :param n_iter:
      number of times to run scipy.minimize

  Returns
  -------
  :return: x_max, The arg max of the acquisition function.
  """
  # Warm up with random points
  x_uniform=[]
  for i in range(bounds.shape[0]):
    num = np.linspace(bounds[i,0],bounds[i,1],num=5,dtype=float)
    lists=[num[1],num[2],num[3]]
    x_uniform.append(lists)
  x_uniforms=[]
  for item in itertools.product(*x_uniform):
    x_uniforms.append(item)
  x_uniforms=np.asarray(x_uniforms)
  n_warmup=n_warmup-x_uniforms.shape[0]
  x_random = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                 size=(n_warmup, bounds.shape[0]))
  x_random = np.asarray(x_random)
  x_tries =np.concatenate((x_uniforms,x_random))
  ys,std= ac(x_tries, gp=gp, y_max=y_max)
  top_10_y = list(map(list(ys).index,heapq.nlargest(10,list(ys))))
  max_information = []
  x_view = bo.space.X
  y_view = bo.space.Y
  length = bo.space._length
  for i in top_10_y:
    bo.space.add_observation(x_tries[i], ys[i])
    bo.gp.fit(bo.space.X, bo.space.Y)
    temp_ys,temp_std = ac(x_tries,gp=gp,y_max=y_max)
    max_information.append(sum(abs(std - temp_std)))
    bo.space._length = length
    bo.space._Xarr = x_view
    bo.space._Yarr = y_view
    bo.space._Xview = x_view
    bo.space._Yview = y_view
  max_information = np.asarray(max_information)
  x_max = x_tries[top_10_y[max_information.argmax()]]
  y_predcit = ys[top_10_y[max_information.argmax()]]
  # x_max=[]
  # x_max.append(xx_max)
  # x_max.append(xx_max)
  # max_acq = ys.max()
  # acs = UtilityFunction(
  #   kind='ucb',
  #   kappa=2.576,
  #   xi=0,
  # ).utility
  # yss=acs(x_max,gp=gp,y_max=y_max)
  # Explore the parameter space more throughly
  # x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
  #                                size=(n_iter, bounds.shape[0]))
  #try to see the result of delete the second find underhere
  # for x_try in x_seeds:
  #   # Find the minimum of minus the acquisition function
  #   #
  #   res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
  #                  x_try.reshape(1, -1),
  #                  bounds=bounds,
  #                  method="L-BFGS-B")
  #
  #   # See if success
  #   if not res.success:
  #     continue
  #
  #   # Store it if better than previous minimum(maximum).
  #   if max_acq is None or -res.fun[0] >= max_acq:
  #    # x_max = res.x
  #     x_max=x_try
  #     max_acq = -res.fun[0]
  #return np.clip(x_tries[ys.argmax()], bounds[:, 0], bounds[:, 1]), yss[0]
  return x_max , y_predcit
class UtilityFunction(object):
  """
  An object to compute the acquisition functions.
  """

  def __init__(self, kind, kappa, xi):
    """
    If UCB is to be used, a constant kappa is needed.
    """
    self.kappa = kappa

    self.xi = xi
    if kind not in ['ucb', 'ei', 'poi']:
      err = "The utility function " \
            "{} has not been implemented, " \
            "please choose one of ucb, ei, or poi.".format(kind)
      raise NotImplementedError(err)
    else:
      self.kind = kind

  def utility(self, x, gp, y_max):
    if self.kind == 'ucb':
      return self._ucb(x, gp, self.kappa)
    if self.kind == 'ei':
      return self._ei(x, gp, y_max, self.xi)
    if self.kind == 'poi':
      return self._poi(x, gp, y_max, self.xi)

  @staticmethod
  def _ucb(x, gp, kappa):
    mean, std = gp.predict(x, return_std=True)
    #v=1
    #delta=0.1
    #d=10
    #t=lens
    #kappa=np.sqrt(v*(2*np.log((t**(d/2.0+2))*(np.pi**2)/(3.*delta))))/4
    return mean + kappa * std , np.asarray(std)

  @staticmethod
  def _ei(x, gp, y_max, xi):
    mean, std = gp.predict(x, return_std=True)

    z = (mean - y_max - xi)/std
    c = (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)
    return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z),np.asarray(std)

  @staticmethod
  def _poi(x, gp, y_max, xi):
    mean, std = gp.predict(x, return_std=True)
    z = (mean - y_max - xi)/std
    return norm.cdf(z)


def unique_rows(a):
  """
  A functions to trim repeated rows that may appear when optimizing.
  This is necessary to avoid the sklearn GP object from breaking

  :param a: array to trim repeated rows from

  :return: mask of unique rows
  """
  if a.size == 0:
    return np.empty((0,))

  # Sort array and kep track of where things should go back to
  order = np.lexsort(a.T)
  reorder = np.argsort(order)

  a = a[order]
  diff = np.diff(a, axis=0)
  ui = np.ones(len(a), 'bool')
  ui[1:] = (diff != 0).any(axis=1)

  return ui[reorder]


def ensure_rng(random_state=None):#
  """
  Creates a random number generator based on an optional seed.  This can be
  an integer or another random state for a seeded rng, or None for an
  unseeded rng.
  """
  if random_state is None:
    random_state = np.random.RandomState()
   # print('see the value of random_state')
   # print(random_state)
  elif isinstance(random_state, int):
    random_state = np.random.RandomState(random_state)
  else:
    assert isinstance(random_state, np.random.RandomState)
  return random_state


class BColours(object):
  BLUE = '\033[94m'
  CYAN = '\033[36m'
  GREEN = '\033[32m'
  MAGENTA = '\033[35m'
  RED = '\033[31m'
  ENDC = '\033[0m'


class PrintLog(object):

  def __init__(self, params):

    self.ymax = None
    self.xmax = None
    self.params = params
    self.ite = 1

    self.start_time = datetime.now()
    self.last_round = datetime.now()

    # sizes of parameters name and all
    self.sizes = [max(len(ps), 7) for ps in params]

    # Sorted indexes to access parameters
    self.sorti = sorted(range(len(self.params)),
                        key=self.params.__getitem__)

  def reset_timer(self):
    self.start_time = datetime.now()
    self.last_round = datetime.now()

  def print_header(self, initialization=True):

    if initialization:
      print("{}Initialization{}".format(BColours.RED,
                                        BColours.ENDC))
    else:
      print("{}Bayesian Optimization{}".format(BColours.RED,
                                               BColours.ENDC))

    print(BColours.BLUE + "-" * (29 + sum([s + 5 for s in self.sizes])) +
          BColours.ENDC)

    print("{0:>{1}}".format("Step", 5), end=" | ")
    print("{0:>{1}}".format("Time", 6), end=" | ")
    print("{0:>{1}}".format("Value", 10), end=" | ")

    for index in self.sorti:
      print("{0:>{1}}".format(self.params[index],
                              self.sizes[index] + 2),
            end=" | ")
    print('')

  def print_step(self, x, y, warning=False):

    print("{:>5d}".format(self.ite), end=" | ")

    m, s = divmod((datetime.now() - self.last_round).total_seconds(), 60)
    print("{:>02d}m{:>02d}s".format(int(m), int(s)), end=" | ")

    if self.ymax is None or self.ymax < y:
      self.ymax = y
      self.xmax = x
      print("{0}{2: >10.5f}{1}".format(BColours.MAGENTA,
                                       BColours.ENDC,
                                       y),
            end=" | ")

      for index in self.sorti:
        print("{0}{2: >{3}.{4}f}{1}".format(
            BColours.GREEN, BColours.ENDC,
            x[index],
            self.sizes[index] + 2,
            min(self.sizes[index] - 3, 6 - 2)
        ),
            end=" | ")
    else:
      print("{: >10.5f}".format(y), end=" | ")
      for index in self.sorti:
        print("{0: >{1}.{2}f}".format(x[index],
                                      self.sizes[index] + 2,
                                      min(self.sizes[index] - 3, 6 - 2)),
              end=" | ")

    if warning:
      print("{}Warning: Test point chose at "
            "random due to repeated sample.{}".format(BColours.RED,
                                                      BColours.ENDC))

    print()

    self.last_round = datetime.now()
    self.ite += 1

  def print_summary(self):
    pass
