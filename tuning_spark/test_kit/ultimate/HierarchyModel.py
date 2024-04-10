import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import BaggingRegressor
from pathlib import Path
import pickle
import yaml



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

# 分层建模
def get_divide_model(samples, current_columns, app_columns, mid_columns, obj_columns):
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

# 使用
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

new_os_setting_path = '/home/hmj/tuning_spark/target/target_spark/new_os_configs_info.yml'
new_app_setting_path = '/home/hmj/tuning_spark/target/target_spark/app_configs_info.yml'
new_os_setting_path = Path(new_os_setting_path)
new_app_setting_path = Path(new_app_setting_path)
os_setting = yaml.load(new_os_setting_path.read_text(), Loader=yaml.FullLoader)
app_setting = yaml.load(new_app_setting_path.read_text(), Loader=yaml.FullLoader)
app_columns = app_setting['app_columns']
apperf_columns = os_setting['app_perf_columns']
obj_columns = ['run_times']
save_columns = app_columns + apperf_columns + obj_columns
# predict model
samples_file_path = "/home/hmj/tuning_spark/target/target_spark/data/initial/sql/join_lhs_samples-2000.csv"
Samples = pd.read_csv(samples_file_path, header=0, encoding="gbk")  # 读取建模用的样本数据
# 分层
get_divide_model(Samples, save_columns, app_columns, apperf_columns, obj_columns)
#使用
# result = use_divide_modes(Samples, save_columns, perf_columns)
