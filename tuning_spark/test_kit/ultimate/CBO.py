import time
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from collections import OrderedDict
from matplotlib import cm
import scipy
import itertools
import time 

from utils_functions import *



def CBO(type_trial, update_model_index, tune_configs_columns, causal_models, model_list, select_BO_models_list,
		            mean_functions_list, var_functions_list,x_dict_mean, x_dict_var,
		            num_trials, exploration_set, manipulative_variables, data_x_list, data_y_list,  best_intervention_value, opt_y, best_variable,
		            dict_ranges, functions,observational_samples, initialbo_samples_num, # coverage_total,
		            graph,num_additional_observations, costs,  full_observational_samples, task = 'min', max_N = 200,
					initial_num_obs_samples =100, num_interventions=10, Causal_prior=False):

	## Initialise dicts to store values over trials and assign initial values
	current_cost = []
	global_opt = []
	current_best_x, current_best_y, dict_interventions = initialise_dicts(causal_models, exploration_set, task)  # x_dict_mean, x_dict_var,
	current_best_y[best_variable].append(opt_y)
	current_best_x[best_variable].append(best_intervention_value)
	# global_opt.append(opt_y)
	current_cost.append(0.)

	## Initialise variables
	observed = 0
	trial_intervened = 0.
	cumulative_cost = 0.
	cumulative_cost_mf = 0.
			
	## Define list to store info
	target_function_list = [None]*len(exploration_set)
	space_list = [None]*len(exploration_set)
	# model_list改成由外部传入，因为推荐配置之后需要实际运行再更新model
	# type_trial改成由外部传入，因为需要整体BO执行进行记录

	## Define intervention function
	for s in range(len(exploration_set)):
		#target_function_list[s]        # target_function_list目标函数入口，在调优中实际运行并不需要
		space_list[s] = Intervention_function(get_interventional_dict(exploration_set[s]),
											  min_intervention=list_interventional_ranges(dict_ranges,exploration_set[s])[0],
											  max_intervention=list_interventional_ranges(dict_ranges,exploration_set[s])[1])
	############################# LOOP
	start_time = time.time()
	for i in range(num_trials):
		## Decide to observe or intervene and then recompute the obs coverage
		# coverage_obs = update_hull(observational_samples, manipulative_variables) #计算当前可调整变量的覆盖率
		# rescale = observational_samples.shape[0]/max_N          #当前样本数/样本总数
		#epsilon_coverage = (coverage_obs/coverage_total)/rescale
		epsilon_coverage = 0.5
		uniform = np.random.uniform(0., 1.)  # 可注释

		## At least observe and interve once
		if i == 0:
			uniform = 0.
		if i == 1:
			uniform = 1.0
		# if i == 0 and len(type_trial) == 0:
		# 	type_trial.append(0)
		defalut_config_opty = observational_samples.iloc[0, -1]
		except_improve = (opt_y - defalut_config_opty)/defalut_config_opty

		# 在初始时或推荐配置执行时间小于default时增加观察样本，以及每一次更新完causal model后需要更新mean和var
		if uniform < epsilon_coverage and (len(type_trial) == 0 or except_improve > 4):
			print('\nOptimization step', i)
			type_trial.append(0)
			## Collect observations and append them to the current observational dataset
			if except_improve > 2 and observational_samples.shape[0] < full_observational_samples.shape[0]:     # (initial_num_obs_samples + observed*num_additional_observations)
				print("Add new observation samples!\n")
				observed = type_trial.count(0)
				new_observational_samples = observe(num_observation = num_additional_observations,
													complete_dataset = full_observational_samples,
													initial_num_obs_samples= initial_num_obs_samples)   #可以定义一个function，再读取20个观察样本
				observational_samples = observational_samples.append(new_observational_samples)
				#observational_samples = pd.concat([observational_samples, new_observational_samples], ignore_index=True)

			## Refit the models for the conditional distributions
			functions = graph.refit_models(observational_samples)

			x_dict_mean = {}
			x_dict_var = {}
			for model in causal_models:
				x_dict_mean[model] = {}
				x_dict_var[model] = {}

			## Update the mean functions and var functions given the current set of observational data. This is updating the prior.
			mean_functions_list, var_functions_list = update_all_do_functions(graph, exploration_set, functions, dict_interventions,
														observational_samples.copy(), x_dict_mean, x_dict_var, tune_configs_columns)
			# 计算mean和var函数时会改变传入的samples,要使用observational_samples.copy()
			
			## Update current optimal solution. If I observe the cost and the optimal y are the same of the previous trial
			# global_opt.append(global_opt[i])
			# current_cost.append(current_cost[i])
		elif uniform >= epsilon_coverage:
			# 实际执行配置
			print('\nOptimization step', i)
			type_trial.append(1)
			trial_intervened += 1
			## When we decid to interve we need to compute the acquisition functions based on the GP models and decide the variable/variables to intervene
			## together with their interventional data

			## Define list to store info
			y_acquisition_list = [None]*len(exploration_set)
			x_new_list = [None]*len(exploration_set)
			
			## This is the global opt from previous iteration
			current_global = find_current_global(current_best_y, dict_interventions, task)  # CBO模型中的最小值

			## If in the previous trial we have observed we want to update all the BO models as the mean functions and var functions computed 
			## via the DO calculus are changed 
			## If in the previous trial we have intervened we want to update only the BO model for the intervention for which we have collected additional data 
			# updata = 0
			# 与causal models一样，只在最初时更新全部model，之后都是选择更新
			if observational_samples.shape[0] ==  initialbo_samples_num :  # initial_num_obs_samples+observed*num_additional_observations:   # and updata == 1:
				print('Updating all model Start!')
				for s in range(len(exploration_set)):
					print('Updating model:', s)
					model_list[s] = update_BO_models(mean_functions_list[s], var_functions_list[s], data_x_list[s], data_y_list[s], Causal_prior)
			else:
				# 要更新的BO model,根据要更新的causal model进行
				print('Updating model:', update_model_index)
				model_list[update_model_index] = update_BO_models(mean_functions_list[update_model_index], var_functions_list[update_model_index],
																  data_x_list[update_model_index], data_y_list[update_model_index], Causal_prior)

			## Compute acquisition function given the updated BO models for the interventional data
			## Notice that we use current_global and the costs to compute the acquisition functions
			print("Updating model end ! \nAcquisition Start !")
			for s in range(len(exploration_set)):
				y_acquisition_list[s], x_new_list[s] = find_next_y_point(space_list[s], model_list[s], current_global, exploration_set[s], costs, task = task)

			## Selecting the variable to intervene based on the values of the acquisition functions
			# var_to_intervene = exploration_set[np.where(y_acquisition_list == np.max(y_acquisition_list))[0][0]] #找出期望改进最大的model/可操纵变量列表
			# index = np.where(y_acquisition_list == np.max(y_acquisition_list))[0][0] #记录其对应的索引
			## hmj
			index, var_to_intervene, model = get_bo_model(select_BO_models_list, y_acquisition_list, exploration_set, causal_models)
			current_best_x[model].append(x_new_list[index][0])
			##
			print('Selected Current recommend model: {}_{}_BO'.format(index, model))
			print('Selected intervention varliable: ', var_to_intervene)
			print('Selected configs: ', x_new_list[index])

			# save to int
			# x_new_list[index] = np.round(x_new_list[index]).astype(int)

			## Evaluate the target function at the new point
			# y_new = target_function_list[index](x_new_list[index])
			# y_new = np.array([[-1]])
			# print('Target function at selected point: ', y_new)

			# ## Append the new data and set the new dataset of the BO model
			# data_x, data_y_x = add_data([data_x_list[index], data_y_list[index]],
			# 									  [x_new_list[index], y_new])
			#
			# data_x_list[index] = np.vstack((data_x_list[index], x_new_list[index]))
			# data_y_list[index] = np.vstack((data_y_list[index], y_new))
			#
			# model_list[index].set_data(data_x, data_y_x)
			#
			# ## Compute cost
			# x_new_dict = get_new_dict_x(x_new_list[index], exploration_set[index])
			# cumulative_cost += total_cost(var_to_intervene, costs, x_new_dict)
			# var_to_intervene = dict_interventions[index]
			# current_cost.append(cumulative_cost)
			#
			#
	 		## Update the dict storing the current optimal solution
			# current_best_x[model].append(x_new_list[index][0])
			# current_best_y[var_to_intervene].append(y_new[0][0])
			#
			#
			# ## Find the new current global optima
			# current_global = find_current_global(current_best_y, dict_interventions, task)
			# global_opt.append(current_global)
			#
			# print('####### Current_global #########', current_global)
			#
			# ## Optimise BO model given the new data
			# model_list[index].optimize()

	## Compute total time for the loop
	total_time = time.time() - start_time

	return (current_best_x, current_best_y, model_list, mean_functions_list, var_functions_list, x_dict_mean, x_dict_var, global_opt, total_time, index)  # , observed


def CameoCBO(type_trial, update_model_index, tune_configs_columns, causal_models, model_list, select_BO_models_list,
		mean_functions_list, var_functions_list, x_dict_mean, x_dict_var,
		num_trials, exploration_set, manipulative_variables, data_x_list, data_y_list, best_intervention_value, opt_y,best_variable,
		dict_ranges, functions, observational_samples, initialbo_samples_num,  # coverage_total,
		graph, num_additional_observations, costs, full_observational_samples, task='min', max_N=200,
		initial_num_obs_samples=100, num_interventions=10, Causal_prior=False):
	## Initialise dicts to store values over trials and assign initial values
	current_cost = []
	global_opt = []
	current_best_x, current_best_y, dict_interventions = initialise_dicts(causal_models, exploration_set, task)  # x_dict_mean, x_dict_var,
	current_best_y[best_variable].append(opt_y)
	current_best_x[best_variable].append(best_intervention_value)
	# global_opt.append(opt_y)
	current_cost.append(0.)

	## Initialise variables
	observed = 0
	trial_intervened = 0.
	cumulative_cost = 0.
	cumulative_cost_mf = 0.

	## Define list to store info
	target_function_list = [None] * len(exploration_set)
	space_list = [None] * len(exploration_set)
	# model_list改成由外部传入，因为推荐配置之后需要实际运行再更新model
	# type_trial改成由外部传入，因为需要整体BO执行进行记录

	## Define intervention function
	for s in range(len(exploration_set)):
		# target_function_list[s]        # target_function_list目标函数入口，在调优中实际运行并不需要
		space_list[s] = Intervention_function(get_interventional_dict(exploration_set[s]),min_intervention=list_interventional_ranges(dict_ranges, exploration_set[s])[0],
											  max_intervention=list_interventional_ranges(dict_ranges, exploration_set[s])[1])
	############################# LOOP
	start_time = time.time()
	for i in range(num_trials):
		## Decide to observe or intervene and then recompute the obs coverage
		# coverage_obs = update_hull(observational_samples, manipulative_variables) #计算当前可调整变量的覆盖率
		# rescale = observational_samples.shape[0]/max_N          #当前样本数/样本总数
		# epsilon_coverage = (coverage_obs/coverage_total)/rescale
		epsilon_coverage = 0.5
		uniform = np.random.uniform(0., 1.)  # 可注释

		## At least observe and interve once
		if i == 0:
			uniform = 0.
		if i == 1:
			uniform = 1.0
		# if i == 0 and len(type_trial) == 0:
		# 	type_trial.append(0)
		defalut_config_opty = observational_samples.iloc[0, -1]
		except_improve = (opt_y - defalut_config_opty) / defalut_config_opty

		# 在初始时或推荐配置执行时间小于default时增加观察样本，以及每一次更新完causal model后需要更新mean和var
		if uniform < epsilon_coverage and (len(type_trial) == 0 or except_improve > 4):
			print('\nOptimization step', i)
			type_trial.append(0)
			## Collect observations and append them to the current observational dataset
			if except_improve > 2 and observational_samples.shape[0] < full_observational_samples.shape[
				0]:  # (initial_num_obs_samples + observed*num_additional_observations)
				print("Add new observation samples!\n")
				observed = type_trial.count(0)
				new_observational_samples = observe(num_observation=num_additional_observations,complete_dataset=full_observational_samples,initial_num_obs_samples=initial_num_obs_samples)  # 可以定义一个function，再读取20个观察样本
				observational_samples = observational_samples.append(new_observational_samples)
			# observational_samples = pd.concat([observational_samples, new_observational_samples], ignore_index=True)

			## Refit the models for the conditional distributions
			functions = graph.refit_models(observational_samples)

			x_dict_mean = {}
			x_dict_var = {}
			for model in causal_models:
				x_dict_mean[model] = {}
				x_dict_var[model] = {}

			## Update the mean functions and var functions given the current set of observational data. This is updating the prior.
			mean_functions_list, var_functions_list = update_all_do_functions(graph, exploration_set, functions, dict_interventions,
																			  observational_samples.copy(), x_dict_mean,x_dict_var, tune_configs_columns)
		# 计算mean和var函数时会改变传入的samples,要使用observational_samples.copy()

		## Update current optimal solution. If I observe the cost and the optimal y are the same of the previous trial
		# global_opt.append(global_opt[i])
		# current_cost.append(current_cost[i])
		elif uniform >= epsilon_coverage:
			# 实际执行配置
			print('\nOptimization step', i)
			type_trial.append(1)
			trial_intervened += 1
			## When we decid to interve we need to compute the acquisition functions based on the GP models and decide the variable/variables to intervene
			## together with their interventional data

			## Define list to store info
			y_acquisition_list = [None] * len(exploration_set)
			x_new_list = [None] * len(exploration_set)

			## This is the global opt from previous iteration
			current_global = find_current_global(current_best_y, dict_interventions, task)  # 几个BO模型中的最小值

			## If in the previous trial we have observed we want to update all the BO models as the mean functions and var functions computed
			## via the DO calculus are changed
			## If in the previous trial we have intervened we want to update only the BO model for the intervention for which we have collected additional data
			# updata = 0
			# 与causal models一样，只在最初时更新全部model，之后都是选择更新
			if observational_samples.shape[0] == initialbo_samples_num:  # initial_num_obs_samples+observed*num_additional_observations:   # and updata == 1:
				print('Updating all model Start!')
				for s in range(len(exploration_set)):
					print('Updating model:', s)
					model_list[s] = update_BO_models(mean_functions_list[s], var_functions_list[s], data_x_list[s],
													 data_y_list[s], Causal_prior)
			else:
				# 要更新的BO model,根据要更新的causal model进行
				for updata_index in range(update_model_index):
					print('Updating model:', updata_index)
					model_list[updata_index] = update_BO_models(mean_functions_list[updata_index],var_functions_list[updata_index],
																	  data_x_list[updata_index],data_y_list[updata_index], Causal_prior)

			## Compute acquisition function given the updated BO models for the interventional data
			## Notice that we use current_global and the costs to compute the acquisition functions
			print("Updating model end ! \nAcquisition Start !")
			for s in range(len(exploration_set)):
				y_acquisition_list[s], x_new_list[s] = find_next_y_point(space_list[s], model_list[s], current_global, exploration_set[s], costs, task=task)

			## Selecting the variable to intervene based on the values of the acquisition functions
			# var_to_intervene = exploration_set[np.where(y_acquisition_list == np.max(y_acquisition_list))[0][0]] #找出期望改进最大的model/可操纵变量列表
			# index = np.where(y_acquisition_list == np.max(y_acquisition_list))[0][0] #记录其对应的索引
			## hmj
			config_acquistion = []
			config_list = []
			causal_model_list = []
			config_num = data_x_list[0].shape[0]
			for i in range(config_num):
				current_config_acquistion = caculate_acquisition_vaule(space_list[0], model_list[0], data_x_list[0][i,None,:] ,current_global,
																	   exploration_set[0], costs, task=task)
				config_list.append(data_x_list[0][i,None,:])
				config_acquistion.append(current_config_acquistion)
				causal_model_list.append(causal_models[0])
			config_acquistion += y_acquisition_list
			config_list += x_new_list
			causal_model_list += causal_models
			#选warm中预期改进大的
			max_acquistion_index = np.where(config_acquistion[:config_num+1] == np.max(config_acquistion[:config_num+1]))[0][0]
			#如果没有相同的参数就保持
			common_par_list = []
			for key in exploration_set[1]:
				if key in exploration_set[0]:
					common_par_list.append(key)
			if len(common_par_list) != 0:
				for config_index in range(config_num+1):
					# 计算跟最佳的距离
					if config_acquistion[max_acquistion_index] - config_acquistion[config_index] < 0.1:
						# 距离小就计算其在cold的预期改进
						copy_new_list = x_new_list[1].copy()
						# 复制参数值
						for key in common_par_list:
							par_index = exploration_set[1].index(key)
							copy_new_list[0][par_index] = config_list[config_index][0][exploration_set[0].index(key)]
						for par_key in exploration_set[1]:
							if par_key not in common_par_list:
								copy_new_list[0][exploration_set[1].index(par_key)] = dict_ranges[par_key][0]
						# 所有配置的预期改进表
						config_list[config_index] = copy_new_list.copy()
						config_acquistion[config_index] = caculate_acquisition_vaule(space_list[1], model_list[1], config_list[config_index] ,current_global,
																	   exploration_set[1], costs, task=task)
						causal_model_list[config_index] = causal_models[1]
			# 选择所有推荐配置中预期改进最大的
			max_acquistion_index = np.where(config_acquistion[:config_num+1] == np.max(config_acquistion[:config_num+1]))[0][-1]
			model = causal_model_list[max_acquistion_index]
			index = causal_models.index(model)#max_acquistion_index
			var_to_intervene = exploration_set[index]
			current_best_x[model].append(config_list[max_acquistion_index][0]) #x_new_list[index][0]

			# index, var_to_intervene, model = get_bo_model(select_BO_models_list, y_acquisition_list, exploration_set, causal_models)
			# current_best_x[model].append(x_new_list[index][0])
			##
			print('Selected Current recommend model: {}_{}_BO'.format(index, model))
			print('Selected intervention varliable: ', var_to_intervene)
			print('Selected configs: ', config_list[max_acquistion_index]) #

	## Compute total time for the loop
	total_time = time.time() - start_time

	return (current_best_x, current_best_y, model_list, mean_functions_list, var_functions_list, x_dict_mean, x_dict_var,
	global_opt, total_time, index)  # , observed
