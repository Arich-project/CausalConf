import os
import sys
from datetime import datetime
line_all = []
str_key = []
str_value = []

with open('/home/hmj/sd_hibench/spark/spark-bopp-test/hibench/report/hibench.report') as f:
    for line in f:
        # print(type(line), '\n')
        line_all.append(line)
str_key = line_all[0].split()
str_value = line_all[1].split()

curr_time=datetime.now()
time=curr_time.strftime("%m-%d")

# read outcome and to caculate
with open('/home/hmj/sd_hibench/spark/spark-bopp-test/hibench/report/hibench', 'a+') as f2:
    for i in range(len(str_key)):
        f2.writelines(str_key[i]+', '+str_value[i]+os.linesep)

# create file with put the result
file_path=sys.argv[2].split('/')   # sys.argv : the parameters from outside  # use '/' to split list like [sql,aggregation]
path='/home/hmj/tuning_spark/target/target_spark/results/run_time/'+file_path[0]  # the outcome every times to save
isExists=os.path.exists(path)
if not isExists:
    os.makedirs(path)
file_path=path+'/'+file_path[1]
isExists=os.path.exists(file_path)
if not isExists:
    with open(file_path,'a+') as f3:
        str = 'experiment       type                        date             time        input_data_size     duration(s)     throughput(byte/s)  throughput/node   '
        f3.writelines(str + os.linesep)
with open(file_path,'a+') as f3:
    f3.writelines(sys.argv[1]+': '+line_all[1]+os.linesep)
# the outcome current times
path2 = '/home/hmj/tuning_spark/target/target_spark/results/temp_runtime'
with open(path2,'w') as f4:
    f4.writelines(sys.argv[1]+': '+line_all[1]+os.linesep)

