from .bo import BOBayesianOptimizer
from .bo_dropout import BOdropoutOptimizer
from .bo_all_pp import BOallppBayesianOptimizer
from .bo_half_pp import BOhalfppBayesianOptimizer
from .bo_bliss import BOblissBayesianOptimizer
from .bo_pp_replace import BOreplaceBayesianOptimizer
from .rand import  RandOptimizer
from .restune import RestuneBayesianOptimizer
available_optimizer = [
    'bo',
    'bo_halfpp',
    'bo_allpp',
    'bodropout',
    'bonew',
    'bobliss',
    'bopp_replace',
    'rand',
    'restune'
]
def create_optimizer(name, configs, extra_vars={}):
  assert name in available_optimizer, f'optimizer [{name}] not supported.'
  if name == 'bo':
    return BOBayesianOptimizer(configs, bo_conf=extra_vars)
  elif name == 'bo_halfpp':
    return BOhalfppBayesianOptimizer(configs,bo_conf=extra_vars)
  elif name == 'bo_allpp':
    return BOallppBayesianOptimizer(configs,bo_conf=extra_vars)
  elif name == 'bodropout':
    return BOdropoutOptimizer(configs, dropout_conf=extra_vars)
  elif name == 'bobliss':
    return BOblissBayesianOptimizer(configs,extra_vars)
  elif name == 'bopp_replace':
    return BOreplaceBayesianOptimizer(configs,extra_vars)
  elif name == 'restune':
    return RestuneBayesianOptimizer(configs,extra_vars)
  elif name == 'rand':
    return RandOptimizer(configs)
