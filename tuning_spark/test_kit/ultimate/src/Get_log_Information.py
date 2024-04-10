import numpy as np
import os
import json
from statistics import mean
def extract_log_information(file):
    runtime = []
    # get the value of keyword "Task Metrics"
    with open(file) as f:
        i = 0
        for line in f.readlines():
            line_dict = json.loads(line)
            if line_dict['Event'] == "SparkListenerTaskEnd":
                runtime.append(line_dict['Task Metrics'])
            # get the type of task
            if line_dict['Event'] == "SparkListenerStageSubmitted":
                if line_dict["Stage Info"]["Stage ID"]==0:
                    type_of_event=line_dict["Stage Info"]["Stage Name"]
                    type_of_event=type_of_event.split()[0]
    #   for i in len(runtime):
    # set the name of feature_vactor
    all_imformation = {}
    feature1 = ["Executor Deserialize Time", "Executor Deserialize CPU Time", "Executor Run Time", "Executor CPU Time",
                "Result Size",
                "JVM GC Time", "Result Serialization Time", "Memory Bytes Spilled", "Disk Bytes Spilled"]
    feature2 = ["Remote Blocks Fetched", "Local Blocks Fetched", "Fetch Wait Time", "Remote Bytes Read",
                "Remote Bytes Read To Disk",
                "Local Bytes Read", "Total Records Read"]
    information_numeric=[]
    #取出feature1中对应的value
    for each_features in feature1:
        information = []
        for num in runtime:
            if each_features in num:
                information.append(num[each_features])
        information_numeric.append(information)
        imformations = " ".join(str(i) for i in information)
        all_imformation[each_features] = imformations
    runtime2 = []
    #取出Shuffle Read Metrics下对应的几个特征feature2
    for line in runtime:
        if "Shuffle Read Metrics" in line:
            runtime2.append(line['Shuffle Read Metrics'])
        else:
            runtime2.append(' ')
    #取出feature2中对应的value
    for each_features in feature2:
        information = []
        for num in runtime2:
            if each_features in num:
                information.append(num[each_features])
        information_numeric.append(information)
        imformations = " ".join(str(i) for i in information)
        all_imformation[each_features] = imformations
    #把取出的所有特征对应的数据存入文件中
    with open(file+'_information', 'a+') as f1:
        f1.writelines('the type of the task:'+type_of_event+os.linesep)
        for k, v in all_imformation.items():
            f1.writelines("{:<30}".format(k))
            f1.writelines("{:<10}".format(v) + os.linesep)
    feature=feature1+feature2
    feature_vector=[]
    #计算每个特征对应的max,min,mean,std
    for data in information_numeric:
        if(data==[]):
            maxdata=0
            mindata=0
            meandata=0
            standard_deviation=0
        else:
            maxdata=np.max(data)
            mindata=np.min(data)
            meandata=np.mean(data)
            standard_deviation=np.std(data)
        temp=[maxdata,mindata,meandata,standard_deviation]
        feature_vector.append(temp)
    #print('the feature_vector of this task:')
    #save the (max,min,mean,std) into file
    with open(file+'feature_vector', 'a+') as f2:
        f2.writelines("{:<30}".format('the name of feature')+"{:<20}".format('max')+"{:<20}".format('min')+
                      "{:<20}".format('mean')+"{:<20}".format('std'))
        for k,v in zip(feature,feature_vector):
            f2.writelines("{:<30}".format(k))
            for i in v:
                f2.writelines("{:<20}".format(i))
            f2.writelines(os.linesep)


