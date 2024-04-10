import os
import sys
import numpy as np
import json
from statistics import mean
from datetime import datetime
line_all = []
str_key = []
str_value = []
curr_time=datetime.now()
time=curr_time.strftime("%m-%d")
file='/home/hmj/tuning_spark/target/target_spark/event_logs/'+sys.argv[2]+'/'+sys.argv[1]
def extract_log_information(file):
    TaskMetrics = []
    #get the value of keyword "Task Metrics"
    with open(file) as f:
        i = 0
        for line in f.readlines():
            line_dict = json.loads(line)
            if line_dict['Event'] == "SparkListenerTaskEnd":
                if 'Task Metrics' in line_dict:   # hmj # hava ‘SparkListenerTaskEnd’ but hava not  'Task Metrics'
                    TaskMetrics.append(line_dict['Task Metrics'])
            # get the type of task
            if line_dict['Event'] == "SparkListenerStageSubmitted":
                if line_dict["Stage Info"]["Stage ID"]==0:
                    type_of_event=line_dict["Stage Info"]["Stage Name"]
                    type_of_event=type_of_event.split()[0]
    # {"Executor Deserialize Time": 28128, "Executor Deserialize CPU Time": 1286694897, "Executor Run Time": 1167,
    #  "Executor CPU Time": 337188736, "Result Size": 2858, "JVM GC Time": 141, "Result Serialization Time": 1,
    #  "Memory Bytes Spilled": 0, "Disk Bytes Spilled": 0,
    #  "Shuffle Read Metrics": {"Remote Blocks Fetched": 0, "Local Blocks Fetched": 0, "Fetch Wait Time": 0,
    #                           "Remote Bytes Read": 0, "Local Bytes Read": 0, "Total Records Read": 0},
    #  "Shuffle Write Metrics": {"Shuffle Bytes Written": 156495, "Shuffle Write Time": 22938712,
    #                            "Shuffle Records Written": 4932},
    #  "Input Metrics": {"Bytes Read": 915674, "Records Read": 4932},
    #  "Output Metrics": {"Bytes Written": 0, "Records Written": 0},
    #  "Updated Blocks": [{"Block ID": "broadcast_0_piece0","Status": {"Storage Level": {"Use Disk": false,"Use Memory": true,"Deserialized": false,"Replication": 1},"Memory Size": 25913,"Disk Size": 0}},
    #                     {"Block ID": "broadcast_0","Status": {"Storage Level": {"Use Disk": false,"Use Memory": true,"Deserialized": true,"Replication": 1},"Memory Size": 408072,"Disk Size": 0}}]
    #  }
    # set the name of feature_vactor
    Result_metric=[]
    events_key = []
    for TaskMetric in TaskMetrics:
        Row = []
        for key in TaskMetric:
            if key == "Shuffle Read Metrics":
                for keys in TaskMetric[key]:
                    if keys not in events_key:
                        events_key.append(keys)
                    Row.append(TaskMetric[key][keys])
            elif key == "Shuffle Write Metrics":
                for keys in TaskMetric[key]:
                    if keys not in events_key:
                        events_key.append(keys)
                    Row.append(TaskMetric[key][keys])
            elif key == "Input Metrics":
                for keys in TaskMetric[key]:
                    if keys not in events_key:
                        events_key.append(keys)
                    Row.append(TaskMetric[key][keys])
            elif key == "Output Metrics":
                for keys in TaskMetric[key]:
                    if keys not in events_key:
                        events_key.append(keys)
                    Row.append(TaskMetric[key][keys])
            elif key == "Updated Blocks":
                continue
            else:
                if key not in events_key:
                    events_key.append(key)
                Row.append(TaskMetric[key])
        Result_metric.append(Row)
    Result_metric = np.asarray(Result_metric)
    feature_vector = []
    for i in range(Result_metric.shape[1]):
        data = Result_metric[:, i]
        maxdata = np.max(data)
        mindata = np.min(data)
        meandata = np.mean(data)
        standard_deviation = np.std(data)
        temp = [maxdata, mindata, meandata, standard_deviation]
        feature_vector.append(temp)
    feature_vector = np.asarray(feature_vector)

    file_path=sys.argv[2].split('/')
    path1 = '/home/hmj/tuning_spark/target/target_spark/results/feature_vector/' + file_path[0]
    isExists = os.path.exists(path1)
    if not isExists:
        os.makedirs(path1)
    i = 0
    with open(path1+'/'+'{}_log'.format(file_path[1]), 'a+') as f:
        f.writelines('{:<15} '.format(sys.argv[1]))
        f.writelines('{:<25}'.format(type_of_event))
        for tol in feature_vector:
            f.writelines(" {:<20} ".format(events_key[i].replace(" ", "_")))
            for part in tol:
                f.writelines("{:<20}".format('%.4f' %part))
                f.writelines("  ")
            i += 1
        f.writelines(os.linesep)

    path2 = '/home/hmj/tuning_spark/target/target_spark/results/temp_feature_vector'
    i = 0
    with open(path2,'w') as f:
        f.writelines('{:<25}'.format(type_of_event))
        for tol in feature_vector:
            f.writelines("{:<20} ".format(events_key[i].replace(" ", "_")))
            for part in tol:
                f.writelines("{:<20}".format('%.4f' %part))
                f.writelines("  ")
            i+=1
        f.writelines(os.linesep)
extract_log_information(file)


