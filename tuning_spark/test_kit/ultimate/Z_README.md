run.py 运行hdconfigor的正式文件
run-rembo.py 测试hdconfigor中新设计的优化算法使用，注释掉了实际在tester和testee上部署运行测试的过程，里面的测试结果用随机数代替
bo-server.yml中存储对于服务器名称，以及所选择算法的配置
python3 run.py ../../target/elasticsearch/tests/bo-server.yml task_name=bo-test exist=delete

aggregate中主义正则表达式，rally不同的track可能是index也可能是index-append。
aggregate-rally-rep3.py 重复三次实验时统计rally的测试结果，引入了最大最小值的统计
aggregate-rally.py 统计error rate 统计三次每次的值，具体可以看注释
aggregate.py 最原始的统计，根据不同的测试软件修改
python3 aggregate-ycsb.py aggregate/agg_ycsb_result-redis.yml task_name=redis-bo-test out=redis-bo-test.csv


YCSB测试的负载在workload/目录下，负载编写后放在该目录下，在tests/目录下的配置文件中指定负载名称，对应该文件，比如stress

tests/bo-ei.yml 这类文件的clients设置要小心，16没问题，1024会挂掉

app_configs_info.yml文件存储app和JVM参数，
os_configs_info.yml文件存储os参数，这两个文件名是在run.py中读取配置变量时写死的，要注意！！！
