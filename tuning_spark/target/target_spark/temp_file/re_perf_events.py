import os
import sys

file = '/home/hmj/tuning_spark/target/target_spark/event_perf/'+sys.argv[2]+'/'+sys.argv[1]
output = list()
with open(file, 'r') as f:
    for line in f:
        output.append(line.split(' '))
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
        perf_output[line[1]] = line[0].replace(',', '')
perf_key = list(perf_output.keys())
perf_value = list(perf_output.values())

file_path=sys.argv[2].split('/')
path1 = '/home/hmj/tuning_spark/target/target_spark/results/feature_vector/' + file_path[0]
isExists = os.path.exists(path1)
if not isExists:
    os.makedirs(path1)
with open(path1+'/'+'{}_perf'.format(file_path[1]), 'a+') as f:
    f.writelines('{:<15}: '.format(sys.argv[1]))
    for i in range(len(perf_key)):
        f.writelines(perf_key[i]+' '+perf_value[i] +'{:5}'.format(' '))
    f.writelines(os.linesep)

