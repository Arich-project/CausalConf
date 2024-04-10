import os
import time
import sys
import pandas as pd
import pydot
import traceback
import numpy as np
import matplotlib.pyplot as plt
from ananke.graphs import ADMG
from networkx import *   # DiGraph, spring_layout, draw
from collections import defaultdict
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci,get_color_edges
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import *
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from ananke.estimation import CausalEffect as CE
from pyCausalFS.GSL.MMHC.MMHC import *   # cmd run
# from test_kit.ultimate.pyCausalFS.GSL.MMHC.MMHC import *   # pycharm run
from castle.algorithms import *
from castle.common import GraphDAG
from random import randint, choice
import random
from collections import OrderedDict
from pyDOE import *
from .deel_data import *

def run_pc_loop(CM, df, tabu_edges, columns, obj_columns,
                 NUM_PATHS):
    """This function is used to run pc in a loop"""
    colmap = {}
    for i in range(len(columns)):
        colmap[i] = columns[i]
    nodes = []
    for i in range(0, len(df.columns)):
        node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        nodes.append(node)

    bk = BackgroundKnowledge()
    for ce in tabu_edges:
        f = list(colmap.keys())[list(colmap.values()).index(ce[0])]
        s = list(colmap.keys())[list(colmap.values()).index(ce[1])]
        bk.add_forbidden_by_node(nodes[f], nodes[s])
    try:
        cg = pc(np.array(df),  0.5, fisherz, verbose=False, background_knowledge=bk)  # G, edges # 0.2
    except:
        cg = pc(np.array(df),  0.4, fisherz, verbose=False)  # G, edges

    edges = get_color_edges(cg.G)
    pc_edges = []
    for edge in edges:
        pc_edges.append(str(edge))

    edges = []
    # resolve notears_edges and fci_edges and update  #使用熵定向策略消除不确定的边
    di_edges, bi_edges = CM.resolve_edges(edges, pc_edges, columns,
                                          tabu_edges, NUM_PATHS, obj_columns)

    # construct mixed graph ADMG   # 容易error，可以先取出全部因果路径后再构图
    # try:
    #     G = ADMG(columns, di_edges=di_edges, bi_edges=bi_edges)
    # except:
    #     print("Construct G Error!")   ##
    #     G = cg.G

    print("--------------------------------------------------------------")
    print("Connections discovered by the causal graph")
    print(di_edges)
    print("--------------------------------------------------------------")

    return cg.G, di_edges, bi_edges

def run_fci_loop(CM, df, tabu_edges, columns, obj_columns,
                 NUM_PATHS):
    """This function is used to run fci in a loop"""
    # NOTEARS causal model hyperparmas
    # _, notears_edges = CM.learn_entropy(df, tabu_edges, 0.75)
    # get bayesian network from DAG obtained by NOTEARS
    # bn = BayesianNetwork(sm)
    fci_edges = CM.learn_fci(df, tabu_edges)
    edges = []
    # resolve notears_edges and fci_edges and update  #使用熵定向策略消除不确定的边
    di_edges, bi_edges = CM.resolve_edges(edges, fci_edges, columns,
                                          tabu_edges, NUM_PATHS, obj_columns)
    # construct mixed graph ADMG
    # G = ADMG(columns, di_edges=di_edges, bi_edges=bi_edges)

    print("--------------------------------------------------------------")
    print("Connections discovered by the causal graph")
    print(di_edges)
    print("--------------------------------------------------------------")

    return di_edges, bi_edges  # G,

def run_ges_loop(df, tabu_edges, columns, obj_columns):
    """This function is used to run ges in a loop"""

    print("GES START!")
    ges_results = ges(np.array(df), score_func='local_score_BIC')
    G = ges_results['G']
    # graph_edges = G.get_graph_edges()
    # print(G)
    gmatrix = G.graph
    GM = Graph_Matrix(columns, gmatrix)
    orig_di_edges, orig_bi_edges, num_graph = GM.getGESEdges()

    di_edges = []
    bi_edges = []
    for edges in orig_di_edges:
        # if edges[0] == obj_columns[0]:
        #     di_edges.append(edges) # di_edges.append((edges[1], edges[0]))
        if edges in tabu_edges:
            continue
        di_edges.append(edges)
    for edges in orig_bi_edges:
        if edges in tabu_edges:
            continue
        bi_edges.append(edges)

    print("--------------------------------------------------------------")
    print("Connections discovered by the causal graph")
    print(di_edges)
    print("--------------------------------------------------------------")

    return G, di_edges, bi_edges

def run_mmhc_loop(CM, df, tabu_edges, columns, obj_columns):
    """This function is used to run mmhc in a loop"""

    mmhc_results = MMHC(df, alpha= 0.15, score_function='bic')     #  ,alpha= 0.15  0.02     # , score_function='bic'
    GM = Graph_Matrix(columns, mmhc_results[0])
    orig_di_edges, orig_bi_edges, num_graph = GM.getmmhcEdges()

    di_edges = []
    bi_edges = []
    for edges in orig_di_edges:
        # if edges[0] == obj_columns[0]:
        #     di_edges.append(edges) #di_edges.append((edges[1], edges[0]))
        if edges in tabu_edges:
            continue
        di_edges.append(edges)
    for edges in orig_bi_edges:
        if edges in tabu_edges:
            continue
        bi_edges.append(edges)

    # construct mixed graph ADMG
    # G = ADMG(columns, di_edges=G_di_edges)  # , bi_edges=bi_edges

    print("--------------------------------------------------------------")
    print("Connections discovered by the causal graph")
    print(di_edges+bi_edges)
    print("--------------------------------------------------------------")

    return  di_edges, bi_edges  # G,

def run_notears_loop(df, tabu_edges, columns, obj_columns):
    record = Notears() #DAG_GNN()  #
    record.learn(np.array(df))
    # GraphDAG(record.causal_matrix)
    GM = Graph_Matrix(columns, record.causal_matrix)
    node_graph, num_graph = GM.getAllEdges()
    # orig_di_edges, , num_graph = GM.getmmhcEdges()
    orig_di_edges = node_graph
    orig_bi_edges = []

    di_edges = []
    bi_edges = []
    for edges in orig_di_edges.edges:
        # if edges[0] == obj_columns[0]:
        #     di_edges.append(edges) #di_edges.append((edges[1], edges[0]))
        if edges in tabu_edges:
            continue
        di_edges.append(edges)
    for edges in orig_bi_edges:
        if edges in tabu_edges:
            continue
        bi_edges.append(edges)

    # construct mixed graph ADMG
    G = ADMG(columns, di_edges=di_edges, bi_edges=bi_edges)

    print("--------------------------------------------------------------")
    print("Connections discovered by the causal graph")
    print(di_edges)
    print("--------------------------------------------------------------")

    return G, di_edges, bi_edges
# -------------------------------------------------------------------------------------------------------
#

class CausalModel:
    def __init__(self, columns):
        print("initializing CausalModel class")
        self.colmap={}
        for i in range(len(columns)):
            self.colmap[i] = columns[i]

    def get_tabu_edges(self, columns, options,
                       objectives):
        """This function is used to exclude edges which are not possible"""
        tabu_edges = []
        # constraint on configuration options
        for opt in options:
            for cur_elem in columns:
                if cur_elem != opt:
                    tabu_edges.append((cur_elem, opt))

        # constraints on performance objetcives
        for obj in objectives:
            for cur_elem in columns:
                if cur_elem != obj:
                    tabu_edges.append((obj, cur_elem))

        return tabu_edges

    def learn_fci(self, df, tabu_edges):
        """This function is used to learn model using FCI"""
        # G, edges = fci(df, fisherz, 0.05, verbose=False)
        # nodes = G.get_nodes()
        nodes = []
        for i in range(0, len(df.columns)):
            node = GraphNode(f"X{i + 1}")
            node.add_attribute("id", i)
            nodes.append(node)

        bk = BackgroundKnowledge()
        for ce in tabu_edges:
            f = list(self.colmap.keys())[list(self.colmap.values()).index(ce[0])]
            s = list(self.colmap.keys())[list(self.colmap.values()).index(ce[1])]
            bk.add_forbidden_by_node(nodes[f], nodes[s])

        try:
            G, edges = fci(np.array(df), fisherz, 0.4, verbose=False, background_knowledge=bk)  # 0.2
        except:
            G, edges = fci(np.array(df), fisherz, 0.4, verbose=False)

        fci_edges = []
        for edge in edges:
            fci_edges.append(str(edge))

        # print("--------------------------------------------------------------")
        # print("Origination causal graph")
        # print(fci_edges)
        # print("--------------------------------------------------------------")

        return fci_edges

    def resolve_edges(self, DAG, PAG,
                      columns, tabu_edges, num_paths,
                      objectives):
        """This function is used to resolve fci (PAG) edges"""
        bi_edge = "<->"
        directed_edge = "-->"
        undirected_edge = "o-o"
        trail_edge = "o->"
        #  entropy only contains directed edges.

        options = {}
        for opt in columns:
            options[opt] = {}
            options[opt][directed_edge] = []
            options[opt][bi_edge] = []
        # add DAG edges to current graph
        for edge in DAG:
            if edge[0] or edge[1] is None:
                options[edge[0]][directed_edge].append(edge[1])
        # replace trail and undirected edges with single edges using entropic policy
        for i in range(len(PAG)):
            if trail_edge in PAG[i]:
                PAG[i] = PAG[i].replace(trail_edge, directed_edge)
            elif undirected_edge in PAG[i]:
                PAG[i] = PAG[i].replace(undirected_edge, directed_edge)
            else:
                continue

        # update causal graph edges
        for edge in PAG:
            cur = edge.split(" ")
            if cur[1] == directed_edge:
                node_one = self.colmap[int(cur[0].replace("X", "")) - 1]
                node_two = self.colmap[int(cur[2].replace("X", "")) - 1]
                options[node_one][directed_edge].append(node_two)
            elif cur[1] == bi_edge:
                node_one = self.colmap[int(cur[0].replace("X", "")) - 1]
                node_two = self.colmap[int(cur[2].replace("X", "")) - 1]

                options[node_one][bi_edge].append(node_two)
            else:
                print("[ERROR]: unexpected edges")

        # extract mixed graph edges
        single_edges = []
        double_edges = []
        for i in options:
            options[i][directed_edge] = list(set(options[i][directed_edge]))
            options[i][bi_edge] = list(set(options[i][bi_edge]))
        for i in options:
            for m in options[i][directed_edge]:
                single_edges.append((i, m))
            for m in options[i][bi_edge]:
                double_edges.append((i, m))
        s_edges = list(set(single_edges) - set(tabu_edges))
        single_edges = []
        for e in s_edges:
            if e[0] != e[1]:
                single_edges.append(e)

        #hmj #将路径中前一半的路径的第一个节点直接指向object
        # for i in range(int(len(s_edges) / 2)):
        #     for obj in objectives:
        #         if s_edges[i][0] != s_edges[i][1]:
        #             single_edges.append((s_edges[i][0], obj))
        # double_edges = list(set(double_edges) - set(tabu_edges))
        #
        # print("--------------------------------------------------------------")
        # print("Connections discovered by the causal graph")
        # print(single_edges)
        # print("--------------------------------------------------------------")
        return single_edges, double_edges

    def get_causal_paths(self, columns, edges, objectives):
        """This function is used to discover causal paths from an objective node"""
        CG = Graph(columns)
        causal_paths = {}
        for edge in edges:
            CG.add_edge(edge[1], edge[0])
        for obj in objectives:
            CG.get_all_paths(obj)
            causal_paths[obj] = CG.path

        return causal_paths

    def compute_path_causal_effect(self, df, paths,
                                   G, K):
        """This function is used to compute P_ACE for each path"""
        ace = {}

        for path in paths:
            ace[str(path)] = 0
            for i in range(0, len(path)):
                if i > 0:
                    obj = CE(graph=G, treatment=path[i], outcome=path[0])  # 使用ananke估计平均因果效应
                    ace[str(path)] += obj.compute_effect(df, "eff-apipw")  # computing the effect    "gformula"
                    # try:
                    #     obj = CE(graph=G, treatment=path[i], outcome=path[0])  # 使用ananke估计平均因果效应
                    #     ace[str(path)] += obj.compute_effect(df, "gformula")  # computing the effect
                    # except:
                    #     continue
            # mean
            # ace[str(path)]
        # rank paths and select top K
        try:
            ## hmj
            paths_dict = {k: v for k, v in sorted(ace.items(), key=lambda item: item[1], reverse=True)}
            paths = list(paths_dict.keys())[:K]
            paths = [eval(key) for key in paths]
            ##
        except TypeError:
            pass
        return paths

    def compute_individual_treatment_effect(self, df, paths,
                                            query, obj_columns,
                                            bug_val, config, cfg,
                                            variable_types):
        """This function is used to compute individual treatment effect"""
        from causality.estimation.nonparametric import CausalEffect
        from causality.estimation.adjustments import AdjustForDirectCauses
        from networkx import DiGraph
        ite = []

        objectives = obj_columns
        option_values = cfg # cfg["option_values"][options.hardware]
        adjustment = AdjustForDirectCauses()

        if query == "best":
            bestval = bug_val
        else:
            bestval = (1 - query) * bug_val

        # single objective treatment effect
        columns_effect = {}
        selected_effect = []
        for path in paths:

            cur_g = DiGraph()
            cur_g.add_nodes_from(path)
            cur_g.add_edges_from([(path[j], path[j - 1]) for j in range(len(path) - 1, 0, -1)])

            for i in range(0, len(path)):
                if i > 0:
                    # ## hmj # without spark_task_cpus
                    # if path[i] == 'spark_task_cpus':
                    #     continue
                    # 个体因果效应只计算 参数-性能目标
                    if path[i] in cfg.keys():
                        if len(objectives) < 2:
                            admissable_set = adjustment.admissable_set(cur_g, [path[i]], [path[0]])

                            effect = CausalEffect(df, [path[i]], [path[0]],
                                                  variable_types=variable_types, admissable_set=list(admissable_set))

                            max_effect = -20000
                            # compute effect for each value for the options
                            for val in option_values[path[i]]:

                                x = pd.DataFrame({path[i]: [val], path[0]: [bestval]})

                                cur_effect = effect.pdf(x)

                                ## hmj
                                if cur_effect > -1:
                                    if path[i] not in columns_effect.keys():
                                        columns_effect[path[i]] = [[val, cur_effect]]
                                    else:
                                        if [val, cur_effect] not in columns_effect[path[i]]:
                                            columns_effect[path[i]].append([val, cur_effect])
                                ##

                                if max_effect < cur_effect:
                                    max_effect = cur_effect
                                    ite.append([path[i], val])
                                    selected_effect.append(max_effect)

        ## hmj
        # print(columns_effect)

        # extract max effect
        if len(selected_effect) > 0:
            for key in columns_effect.keys():
                columns_effect[key] = max(columns_effect[key], key=lambda x: x[-1])
                config[key] = columns_effect[key][0]
                # print(key, config[key])
        else:
            print("no change!")
        # extract random effect
        # if (len(selected_effect) > 0):
        #     for key in columns_effect.keys():
        #         if len(option_values[key]) > 2:
        #             sort_list = sorted(columns_effect[key], key=lambda x: x[-1], reverse=True)
        #             columns_effect[key] = sort_list[:2]
        #             config[key] = columns_effect[key][randint(0, 1)][0]
        #             # print(key, config[key])
        #         else:
        #             columns_effect[key] = max(columns_effect[key], key=lambda x: x[-1])
        #             config[key] = columns_effect[key][0]
        #             # print(key, config[key])
        # else:
        #     print("no change!")
        ##

        # if len(columns_effect) >10:
        #     select_columns = sorted(columns_effect.items(), key=lambda x:x[1][-1], reverse=True)
        #     columns_effect = select_columns[:10]
        # with open('/home/hmj/tuning_spark/target/target_spark/results/causal_graph/causal_path', 'a+') as f:
        #     f.writelines(str(columns_effect.keys()) + os.linesep)

        # if(len(selected_effect) > 0):
        #     selected_index = np.argmax(selected_effect)
        #     config[ite[selected_index][0]] = ite[selected_index][1]
        # else:
        #     print("no change!")

        print("--------------------------------------------------")
        print("--------------------------------------------------")
        print("             Recommended Configuration            ")
        print(config)
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        return config


class Graph:
    def __init__(self, vertices):
        # No. of vertices
        self.V = vertices
        # default dictionary to store graph
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def get_all_paths_util(self, u, visited, path):

        visited[u]= True
        path.append(u)
        # If current vertex is same as destination, then print
        if self.graph[u] == []:
            try:
                if self.path:
                    self.path.append(path[:])
            except AttributeError:
                    self.path=[path[:]]
        else:
            for i in self.graph[u]:
                if visited[i]== False:
                    self.get_all_paths_util(i, visited, path)

        # remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[u]= False

    def get_all_paths(self, s):
        # mark all the vertices as not visited
        visited = {}
        for i in self.V:
            visited[i] = False
        # create an array to store paths
        path = []
        # call the recursive helper function to print all paths
        self.get_all_paths_util(s, visited, path)


