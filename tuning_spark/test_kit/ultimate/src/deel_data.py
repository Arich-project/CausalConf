import os
import time
import sys
import pandas as pd
import pydot
import traceback
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import pickle
from networkx import *   # DiGraph, spring_layout, draw
from random import randint, choice
from pyDOE import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import BaggingRegressor
from pathlib import Path
from statistics import mean


# hmj
def get_random(config):
  res = {}
  for k, conf in config.items():
    c_type = conf.get('float')
    c_range = conf.get('range')
    if c_range is None:
      if c_type:
        random_value = random.uniform(conf['min'], conf['max'])  #连续,小数
        res[k] = round(random_value, 2)
      else:
        res[k] = randint(conf['min'], conf['max'])
    else:
        res[k] = c_range[randint(0, len(c_range) - 1)]  # 离散

  return res

#hmj
def get_specific_values(config):
  type = ['default', 'min', 'max', 'mid']
  specific_values = []
  value = []

  for index in range(len(type)):
    if type[index] == 'mid':
      for k, conf in config.items():
        c_type = conf.get('float')   # 小数
        c_range = conf.get('range')
        if c_range is None:
            if c_type is None:
                mid_values = round((conf['min']+conf['max'])/2)
            else:
                mid_values = round((conf['min'] + conf['max']) / 2, 2)
            value.append(mid_values)     # 连续
        else:
            value.append(c_range[int(len(c_range)/2)])  # 离散
    else:
      for k, conf in config.items():
        c_range = conf.get('range')
        if c_range is None or type[index] == 'default':
            value.append(conf[type[index]])
        else:
            if type[index] == 'min':
                value.append(c_range[0])
            else:
                value.append(c_range[len(c_range)-1])
    specific_values.append(value)
    value = []

  return specific_values

def get_lhs_samples(samples_columns, num, samples):
    dimensions = 1
    for key,values in samples_columns.items():
        c_type = values.get('float')
        c_range = values.get('range')
        one_samples = lhs(dimensions, samples=num)
        if c_range is None:
            one_samples = values['min'] + (values['max'] - values['min'])*one_samples  # 连续
            # if c_type is None:
            #     one_samples = np.round(one_samples).astype(int)
            # else:
            one_samples = np.round(one_samples, decimals=2)
        else:
            one_samples = (len(values['range']) - 1) * one_samples       # 离散
            one_samples = np.round(one_samples).astype(int)
            one_samples = np.array([c_range[i[0]] for i in one_samples]).reshape(-1,1)
            # one_samples = randint(0, len(values['range'])).ppf(one_samples)
        # save one time samples
        if list(samples_columns.keys()).index(key) == 0:
            lhc_samples = one_samples.copy()
        else:
            lhc_samples = np.hstack((lhc_samples, one_samples))
    samples = samples + lhc_samples.tolist()
    return samples

#hmj
def get_option_values(tuning_columns, option_values):
    res = {}
    for key in tuning_columns:
        if key in list(option_values.keys()):
            res[key] = option_values[key]
    return res

#hmj
def normal_samples(samples_df, deel_columns):
    normal_df = samples_df.copy()
    for column in samples_df.keys():
        if column in deel_columns:
            for item in range(len(samples_df[[column]])):
                normal_df.at[item, column] = ((normal_df[column][item] - samples_df[column].min())/(samples_df[column].max() - samples_df[column].min())) * 20

    return normal_df

#hmj
def get_ds_samples(all_samples_df, ds_samples_num):
    all_columns = []

    level_samples = [all_samples_df.iloc[i:i+ds_samples_num] for i in range(0, len(all_samples_df), ds_samples_num)]

    for index, filtered_df in enumerate(level_samples):
        # 去除值不变化的中间性能变量
        unique_counts = filtered_df.nunique()
        level_samples[index] = filtered_df.loc[:, unique_counts != 1]
        all_columns.append(level_samples[index].columns.tolist())

    # 同一个负载下保留相同的中间性能变量
    common_elements = set(all_columns[0])
    for sublist in all_columns[1:]:
        common_elements.intersection_update(sublist)
    common_columns = [column for column in all_columns[0] if column in common_elements]# list(common_elements)

    # 保留相同columns的数据
    for index, filtered_df in enumerate(level_samples):
        level_samples[index] = level_samples[index][common_columns]

    return common_columns, level_samples

def get_cbo_samples(level_samples, similar_ds_sort, initial_num_obs_samples):
    observer_samples = level_samples[similar_ds_sort[0]].copy()
    full_observer_samples = level_samples[similar_ds_sort[0]].copy()
    for index in similar_ds_sort[1:]:
        full_observer_samples = pd.concat([full_observer_samples,level_samples[index]], axis=0)
        if observer_samples.shape[0] < initial_num_obs_samples:
            observer_samples = pd.concat([observer_samples,level_samples[index]], axis=0)

    return observer_samples, full_observer_samples
# hmj
def get_similar_ds(input_workload,input_datasize,datasize_list):
    similar_ds = []
    similar_index = []
    ds_num_list = datasize_list[input_workload]
    distances = [abs(input_datasize-num) for num in ds_num_list]
    min_distance = min(distances)
    closest_numbers = [[i, ds_num_list[i]] for i, dist in enumerate(distances) if dist == min_distance]
    # closest_indices = [i for i, dist in enumerate(distances) if dist == min_distance]
    # return closest_numbers, closest_indices
    for closest_num in closest_numbers:
        similar_index.append(closest_num[0])
        similar_ds.append(closest_num[-1])

    similar_ds_sort = sorted(range(len(distances)), key=lambda i: distances[i])

    return similar_ds, similar_index, min_distance, similar_ds_sort

#hmj
def get_one_sample(filename,columns):

    map = {"sched_sched_wakeup_new": "sched:sched_wakeup_new",
           "sched_sched_wakeup": "sched:sched_wakeup",
           "sched_sched_switch": "sched:sched_switch",
           "sched_sched_stat_runtime": "sched:sched_stat_runtime",
           "sched_sched_process_wait": "sched:sched_process_wait",
           # "sched_sched_load_avg_cpu": "sched:sched_load_avg_cpu",
           # "sched_sched_overutilized": "sched:sched_overutilized",
           "raw_syscalls_sys_enter": "raw_syscalls:sys_enter",
           "raw_syscalls_sys_exit": "raw_syscalls:sys_exit",
           }

    output = list()
    with open(filename, 'r') as f:
        for line in f:
            output.append(line.split(' '))
    if len(output) == 0:
        data = [[0] * len(columns)]
        df = pd.DataFrame(data,columns=columns)
        print("Perf Columns Data Empty!")
        return df

    output = output[5:]
    output.pop()
    output.pop()
    output.pop()
    processed_output = [[] for _ in range(len(output))]
    perf_output = {}
    for line in range(len(output)):
        for elem in output[line]:
            if elem != '':
                processed_output[line].append(elem)
    for line in processed_output:
        if len(line) > 2:
            perf_output[line[1]] = [line[0]]

    pdf = pd.DataFrame(perf_output)
    # process perf df
    tmp_columns = list(pdf)
    pdf[tmp_columns] = pdf[tmp_columns].replace({',': ''}, regex=True)
    pdf[tmp_columns] = pdf[tmp_columns].astype(float)
    # convert columns to json readable format
    perf_columns = columns
    df = pd.DataFrame(columns=perf_columns)

    for col in perf_columns:
        try:
            df[col] = pdf[col]
        except KeyError:
            df[col] = pdf[map[col]]

    return df

def get_log_events_samples(filename,columns):
    # read line
    output = list()
    with open(filename, 'r') as f:
        for line in f:
            output.append(line.split(' '))
    # split line , get all element
    processed_output = [[] for _ in range(len(output))]
    for line in range(len(output)):
        output[line] = output[line][1:]
        output[line].pop()
        for elem in output[line]:
            if elem != '':
                processed_output[line].append(elem)
    # get columns element values
    for line_values in processed_output:
        line_values = [line_values[i:i + 5] for i in range(0, len(line_values), 5)]
    all_perf_output = {}
    # mean value
    for line in line_values:
        all_perf_output[line[0]] = line[-2]
    perf_output = {k: float(all_perf_output[k]) for k in columns}
    # save value to float
    df = pd.DataFrame([perf_output])

    return df

def get_perf_samples():
    per_filename = '/home/hmj/tuning_spark/target/target_spark/results/feature_vector/sql/join_perf'
    map = {"raw_syscalls:sys_enter":"raw_syscalls_sys_enter" ,
           "raw_syscalls:sys_exit": "raw_syscalls_sys_exit"
           }
    with open(per_filename, encoding='utf-8') as f:
        all_infor = []
        row_infor = {}
        # k = 0
        for line in f.readlines():
            line = line.split()
            # stage_feature = line[0]
            for i in range(1, len(line)):
                if (i - 1) % 2 != 0:
                    # if k%5 == 0:
                    row_infor[line[i-1]] = round(float(line[i]),3) # [float(line[i])]
                    # else:
                    #     row_infor[line[i-1]].append(float(line[i]))
                # row_infor.append(float(line[i]))
            # if k%5 == 4:
            # for index,values in row_infor.items():
            #     save_index = sorted(range(len(values)), key=lambda k: values[k], reverse=True)[1:-1]  # 去掉最小最大值的events
            #     result_list = [values[i] for i in save_index]
            #     row_infor[index] = mean(result_list) if len(result_list) > 0 else .0
            #     row_infor[index] = round(row_infor[index], 3)
            all_infor.append(row_infor)
            row_infor =  {}
            # k += 1
    # history_perf_data = np.asarray(all_infor)
    mean_df = pd.DataFrame(all_infor) #.from_dict
    for col in map.keys():
        mean_df.rename(columns={col:map[col]}, inplace=True)

    # hmj #test collection perf_columns samples data
    # per_filename = '/home/hmj/tuning_spark/target/target_spark/results/perf_results/'
    # filename_format = '{}_perf_result_{}'
    # temp_df = pd.DataFrame(columns=perf_columns)
    # mean_df = pd.DataFrame(columns=perf_columns)
    # for id in range(35):
    #     for num in range(3):
    #         filename = os.path.join(per_filename, filename_format.format(id, num))
    #         one_df = get_one_sample(filename, perf_columns)
    #         temp_df = pd.concat([temp_df, one_df], axis= 0 )
    #     temp_df = pd.DataFrame(temp_df.mean()).transpose()
    #     mean_df = pd.concat([mean_df,temp_df], axis = 0)
    #     temp_df = pd.DataFrame(columns=temp_df.columns)

    return mean_df

# #hmj
# def get_graph_nodes(causal_edges):
#
#     causal_nodes = []
#     for edge_node in causal_edges:
#         for node in range(0, len(edge_node)):
#             if edge_node[node] not in causal_nodes:
#                 causal_nodes.append(edge_node[node])
#
#     return causal_nodes

def Choose_parameters_of_BaggingRegressor(X_train, X_test, y_train, y_test , save_name):
    min_error = float('inf')
    filename = './predict_models/Z_{}_model.sav'.format(save_name)
    best_estimators = best_features = best_samples = best_bootstrap  = 0
    for n_estimators in [3, 5, 10, 20, 30, 50, 75, 100]:
        for max_features in [1, 3, 5, 10, 15, 20, 30]:
            for max_samples in [1, 3, 4, 5, 6, 7]:
                for best_bootstrap in [True, False]:
                    model = BaggingRegressor(n_estimators = n_estimators, max_features = max_features, max_samples = max_samples,
                                             bootstrap = best_bootstrap).fit(X_train, y_train)
                    error = mean_absolute_percentage_error(y_test, model.predict(X_test))
                    if error < min_error:
                        min_error = error; best_estimators = n_estimators; best_features = max_features; best_samples = max_samples; bootstrap = best_bootstrap
                        print("\nSetting: best_estimators = {} , best_features = {}, best_samples = {}, bootstrap = {}."
                              .format(n_estimators,max_features,max_samples,best_bootstrap))
                        print("the current predict error is {}".format(error))
                        pickle.dump(model, open(filename, 'wb'))
    return

def Choose_opty_of_BaggingRegressor(X_train, X_test, y_train, y_test , save_name):
    min_error = float('inf')
    filename = './predict_models/Z_{}_model.sav'.format(save_name)
    best_estimators = best_features = best_samples = best_bootstrap  = 0
    for n_estimators in [3, 5, 10, 20, 30, 50, 75, 100]:
        for max_features in [1, 3, 5, 10, 14]:
            for max_samples in [1, 3, 4, 5, 6, 7]:
                for best_bootstrap in [True, False]:
                    model = BaggingRegressor(n_estimators = n_estimators, max_features = max_features, max_samples = max_samples,
                                             bootstrap = best_bootstrap).fit(X_train, y_train)
                    error = mean_absolute_percentage_error(y_test, model.predict(X_test))
                    if error < min_error:
                        min_error = error; best_estimators = n_estimators; best_features = max_features; best_samples = max_samples; bootstrap = best_bootstrap
                        print("\nSetting: best_estimators = {} , best_features = {}, best_samples = {}, bootstrap = {}."
                              .format(n_estimators,max_features,max_samples,best_bootstrap))
                        print("the current predict error is {}".format(error))
                        pickle.dump(model, open(filename, 'wb'))
    return

def get_divide_model(samples,current_columns, app_columns, mid_columns, obj_columns):
    for i in range(len(mid_columns)):
        if mid_columns[i] in current_columns:
            X_samples = samples[app_columns]
            Z_samples = samples[mid_columns[i]]
            y_samples = samples[obj_columns[0]]
            # parameter to middle
            print("=================\nSelect mid model {} Sart!\n=================\n".format(i))
            X_train, X_test, Z_train, Z_test = train_test_split(X_samples, Z_samples, test_size=0.3)
            Choose_parameters_of_BaggingRegressor(X_train, X_test, Z_train, Z_test, 'mid_{}'.format(i))
            print("=================\nSelect mid model {} End!\n=================\n".format(i))
    # middle to obj
    Z_samples = samples[mid_columns]
    Z_train, Z_test, y_train, y_test = train_test_split(Z_samples, y_samples, test_size=0.3)
    Choose_opty_of_BaggingRegressor(Z_train, Z_test, y_train, y_test, 'opt')
    print("=================\nSelect opty model End!\n=================\n")

    # X_samples = samples.iloc[:, :30]
    # y_samples = samples.iloc[:, -1]
    # X_train, X_test, y_train, y_test = train_test_split(X_samples, y_samples, test_size=0.3)
    # Choose_parameters_of_BaggingRegressor(X_train, X_test, y_train, y_test)
    # model = BaggingRegressor(n_estimators=10, max_samples=5, max_features=20, bootstrap=True).fit(X_train, y_train)
    # error = mean_absolute_percentage_error(y_test, model.predict(X_test))
    return

def use_divide_modes(samples, current_columns, mid_columns):
    mid_values = []
    for i in range(len(mid_columns)):
        if mid_columns[i] in current_columns:
            filename = './predict_models/Z_mid_{}_model.sav'.format(i)
            model = pickle.load(open(filename, 'rb'))  # 载入离线模型
            mid_values.append(model.predict(samples))
    input_mid_values = pd.DataFrame(mid_values)
    input_mid_values = input_mid_values.transpose()

    filename = './predict_models/Z_opt_model.sav'
    model = pickle.load(open(filename, 'rb'))  # 载入离线模型
    y_pred = model.predict(input_mid_values)

    return y_pred

def translate_array_value(config_samples, values_range):
    res_config = {}
    key_list = list(values_range.keys())
    for key_i in range(len(key_list)):
        res_config[key_list[key_i]] = config_samples[key_i]

    return res_config

def translate_discretize(option_values):
    cut_num = 6
    for key, values in option_values.items():
        if values.get('range') is None:
            value = np.linspace(values['min'], values['max'], cut_num)
            if 'float' not in option_values[key]:
                value = np.round(value).astype(int).tolist()
                value = sorted(list(set(value)))
            else:
                value = np.round(value, 2).tolist()
            option_values[key].pop('min')
            option_values[key].pop('max')
            option_values[key]['range'] = value

    return option_values

# hmj
def translate_configs(condict_setting,trans_config, trans_type):
    # the parameter with string or bool
    bool_list = ['false', 'true']
    trans_colums = {
                    'spark_storage_replication_proactive': bool_list,
                    'spark_broadcast_compress': bool_list,
                    'spark_io_compression_codec': ['lz4', 'lzf', 'snappy'],
                    'spark_rdd_compress': bool_list,
                    'spark_shuffle_spill_compress': bool_list,
                    'spark_speculation': bool_list,
                    'spark_task_reaper_enabled': bool_list,
                    'spark_shuffle_compress': bool_list,
                    'jvm_gc_collect': ['+UseSerialGC', '+UseParallelGC', '+UseConcMarkSweepGC', '+UseG1GC'],
                    'spark_kryo_referenceTracking': bool_list,
                    'spark_memory_offHeap_enabled': bool_list
                    }

    for items in trans_config:
        if items in condict_setting:
            if condict_setting[items].get('range') is None:
                min_num = condict_setting[items]['min']
                max_num = condict_setting[items]['max']
            else:
                min_num = min(condict_setting[items]['range'])
                max_num = max(condict_setting[items]['range'])

            if 'float' in condict_setting[items]:
                trans_config[items] = round(trans_config[items], 2)
                if trans_config[items] < min_num or trans_config[items] > max_num and items not in trans_colums:
                    # the parameter with float of 'range'
                    try:
                        trans_config[items] = condict_setting[items]['range'][int(round(trans_config[items])) % len(condict_setting[items]['range'])]
                    except KeyError:
                        trans_config[items] = condict_setting[items]['min'] + (condict_setting[items]['max'] - condict_setting[items]['min']) * trans_config[items]  # 连续
                continue

            # try:
            trans_config[items] = int(round(trans_config[items]))
            # except TypeError:
            #     print(trans_config)
            #     print(items)
            #     print(trans_config[items])
            if trans_config[items] < min_num or trans_config[items] > max_num:
                if items not in trans_colums:
                    # the parameter with int of 'range'
                    trans_config[items] = condict_setting[items]['range'][trans_config[items] % len(condict_setting[items]['range'])]
                else:
                    trans_config[items] = condict_setting[items]['range'][len(condict_setting[items]['range']) - 1]

        # the parameter with string or bool
        if trans_type==1:
            if items in trans_colums:
                try:
                    trans_config[items] = trans_colums[items][trans_config[items]]
                except IndexError:
                    trans_config[items] = trans_colums[items][len(trans_colums[items])-1]

    return trans_config

#hmj
def get_resolve_edges(current_columns, loop_graph_edges, obj_column):
    # problem: 中间性能变量之间有环的解决办法：去除离性能目标近的节点的出边
    current_G = DiGraph()
    current_G.add_nodes_from(current_columns)
    current_G.add_edges_from(loop_graph_edges)
    cycles = list(nx.simple_cycles(current_G))
    for i in range(0, len(cycles)):
        # caculate distance between cycle node with obj_cloumn
        shortest_distances = {}
        for cycle in cycles:
            for node in cycle:
                if node not in shortest_distances.keys():
                    try:
                        shortest_distances[node] = nx.shortest_path_length(current_G, source=node, target=obj_column)
                    except NetworkXNoPath:  # in cycles but no path with obj column
                        continue
        if len(shortest_distances) == 0:   # all cycles node without path with obj column
            break
        min_distance_node = min(shortest_distances.values())
        closest_node = [node for node,distance in shortest_distances.items() if distance == min_distance_node]
        # delete node with （close node, cycle node)
        for close_node in closest_node:   # min < columns
            for cycle in cycles:
                for node in cycle:
                    if (close_node, node) in current_G.edges:
                        current_G.remove_edge(close_node, node)
        cycles = list(nx.simple_cycles(current_G))
        if len(cycles) == 0:
            break

    resolve_graph_edges = list(current_G.edges)

    return resolve_graph_edges

#hmj
def get_ds_G_paths(CM, ds_graph_edges, current_columns, obj_columns):
    # identify causal paths # 保留每个datasize因果图的与性能目标相关的因果关系
    ds_paths = CM.get_causal_paths(current_columns, ds_graph_edges,  # edges # 得到所有因果路径 [model_name]
                            obj_columns)
    # delete copy column
    all_paths = {}
    result_list = {}
    for obj_column in obj_columns:
        obj_paths = ds_paths[obj_column]  # ds_paths
        for sublist in obj_paths:
            last_elements = sublist[-1]
            if last_elements not in result_list:
                result_list[last_elements] = sublist
            else:
                # 优先保留长的路径
                if len(sublist) > len(result_list[last_elements]):
                    result_list[last_elements] = sublist
        all_paths[obj_column] = list(result_list.values())
        # all_path[G_index][obj_column] = list(result_list.values())
    return all_paths

#hmj
def draw_graph(graph_edges, id):

    graph_nodes=[]
    for edge_node in graph_edges:
        for node in range(0, len(edge_node)):
            if edge_node[node] not in graph_nodes:
                graph_nodes.append(edge_node[node])

    g = DiGraph()
    # draw_graph_nodes = get_graph_nodes(graph_edges)
    g.add_nodes_from(graph_nodes)
    g.add_edges_from(graph_edges)
    # show graph
    plt.figure(figsize=(16, 13))
    pos = circular_layout(g)
    draw_networkx(g, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_color='black',
                  node_size=800, font_size=15, width=2)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/home/hmj/tuning_spark/target/target_spark/results/causal_graph/{}_graph.jpg'.format(id), dpi=400)
    plt.close()

    return g

# matrix to graph
class Graph_Matrix:
    """
    Adjacency Matrix
    """
    def __init__(self, vertices=[], matrix=[]):
        """

        :param vertices:a dict with vertex id and index of matrix , such as {vertex:index}
        :param matrix: a matrix
        """
        self.matrix = matrix
        self.edges_dict = {}  # {(tail, head):weight}
        self.edges_array = []  # (tail, head, weight)
        self.edges_num = []
        self.vertices = vertices
        self.num_edges = 0

        # if provide adjacency matrix then create the edges list
        if len(matrix) > 0:
            if len(vertices) != len(matrix):
                raise IndexError
            #self.edges = self.getAllEdges()
            #self.num_edges = len(self.edges)

        # if do not provide a adjacency matrix, but provide the vertices list, build a matrix with 0
        elif len(vertices) > 0:
            self.matrix = [[0 for col in range(len(vertices))] for row in range(len(vertices))]

        #self.num_vertices = len(self.matrix)

    def isOutRange(self, x):
        try:
            if x >= self.num_vertices or x <= 0:
                raise IndexError
        except IndexError:
            print("节点下标出界")

    def isEmpty(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices == 0

    def add_vertex(self, key):
        if key not in self.vertices:
            self.vertices[key] = len(self.vertices) + 1

        # add a vertex mean add a row and a column
        # add a column for every row
        for i in range(self.getVerticesNumbers()):
            self.matrix[i].append(0)

        self.num_vertices += 1

        nRow = [0] * self.num_vertices
        self.matrix.append(nRow)

    def getVertex(self, key):
        pass

    def add_edges_from_list(self, edges_list):  # edges_list : [(tail, head, weight),()]
        for i in range(len(edges_list)):
            self.add_edge(edges_list[i][0], edges_list[i][1], edges_list[i][2], )

    def add_edge(self, tail, head, cost=0):
        # if self.vertices.index(tail) >= 0:
        #   self.addVertex(tail)
        if tail not in self.vertices:
            self.add_vertex(tail)
        # if self.vertices.index(head) >= 0:
        #   self.addVertex(head)
        if head not in self.vertices:
            self.add_vertex(head)

        # for directory matrix
        self.matrix[self.vertices.index(tail)][self.vertices.index(head)] = cost
        # for non-directory matrix
        # self.matrix[self.vertices.index(fromV)][self.vertices.index(toV)] = \
        #   self.matrix[self.vertices.index(toV)][self.vertices.index(fromV)] = cost

        self.edges_dict[(tail, head)] = cost
        self.edges_array.append((tail, head, cost))
        self.num_edges = len(self.edges_dict)

    def getEdges(self, V):
        pass

    def getVerticesNumbers(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices

    def getAllVertices(self):
        return self.vertices

    def getAllEdges(self):
        # from networkx import DiGraph
        g_node = DiGraph()
        g_num = DiGraph()
        g_node.add_nodes_from(self.vertices)
        for i in range(0,len(self.vertices)):
            self.edges_num.append("X"+str(i)) # (i+1)

        g_num.add_nodes_from(self.edges_num)

        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if 0 < self.matrix[i][j] < float('inf'):
                    g_node.add_edges_from([(self.vertices[i], self.vertices[j])])
                    g_num.add_edges_from([(self.edges_num[i], self.edges_num[j])])
                    #self.edges_dict[self.vertices[i], self.vertices[j]] = self.matrix[i][j]
                    # self.edges_array.append([self.vertices[i], self.vertices[j], self.matrix[i][j]])

        return g_node, g_num

    def getGESEdges(self):
        # from networkx import DiGraph

        # extract mixed graph edges
        single_edges = []
        double_edges = []
        num_describe = []

        for i in range(0, len(self.vertices)):
            self.edges_num.append("X"+str(i+1))

        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if self.matrix[i][j] == -1 and self.matrix[j][i] == 1:
                    single_edges.append((self.vertices[i], self.vertices[j]))
                    num_describe.append(str(self.edges_num[i] + '-->' + self.edges_num[j]))
                elif self.matrix[i][j] == -1 and self.matrix[j][i] == -1:
                    double_edges.append((self.vertices[i], self.vertices[j]))
                    num_describe.append(str(self.edges_num[i] + '<->' + self.edges_num[j]))

        return single_edges, double_edges, num_describe

    def getmmhcEdges(self):

        # extract mixed graph edges
        single_edges = []
        double_edges = []
        num_describe = []

        for i in range(0, len(self.vertices)):
            self.edges_num.append("X"+str(i+1))

        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if self.matrix[i][j] == -1:
                    single_edges.append((self.vertices[i], self.vertices[j]))
                    num_describe.append(str(self.edges_num[i] + '-->' + self.edges_num[j]))
                elif self.matrix[i][j] == 1 and self.matrix[j][i] == 1:
                    double_edges.append((self.vertices[i], self.vertices[j]))
                    num_describe.append(str(self.edges_num[i] + '<->' + self.edges_num[j]))

        return single_edges, double_edges, num_describe

    def __repr__(self):
        return str(''.join(str(i) for i in self.matrix))

    def to_do_vertex(self, i):
        print('vertex: %s' % (self.vertices[i]))

    def to_do_edge(self, w, k):
        print('edge tail: %s, edge head: %s, weight: %s' % (self.vertices[w], self.vertices[k], str(self.matrix[w][k])))

