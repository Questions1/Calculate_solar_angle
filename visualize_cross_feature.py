
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import svm
from functools import reduce

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_plot(data_train):
    data_train = data_train[data_train['实际功率'] > 0.03 * data_train['实际功率'].max()]
    features = data_train.columns.drop(['时间', '实际功率'])
    fig = plt.figure(figsize=(20, 10))
    for i in range(len(features)):
        ax_i = fig.add_subplot(3, 6, i+1)
        ax_i.scatter(data_train[features[i]].values, data_train['实际功率'].values, s=2, alpha=0.2)
        ax_i.set_xlabel(features[i], fontsize=20)
        ax_i.set_ylabel('实际功率')
    plt.show()


def visualize_solar_data():
    fig = plt.figure(figsize=(20, 10))
    plt.scatter(train_1_new['hours_float'].values, train_1_new['azimuth'].values, s=2, alpha=0.2)
    plt.title('这张图展示了一天中时间与太阳方位的关系')

    # 下面
    fig = plt.figure(figsize=(20, 10))
    plt.scatter(train_1_new['hours_float'].values, train_1_new['altitude'].values, s=2, alpha=0.2)
    plt.title('这张图展示了一天中时间与太阳仰角的关系')

    fig = plt.figure(figsize=(20, 10))
    plt.scatter(train_1_new['yday'].values, train_1_new['azimuth'].values, s=2, alpha=0.2)
    plt.title('这张图展示了一年到头来每天太阳方位的范围变化情况')

    fig = plt.figure(figsize=(20, 10))
    plt.scatter(train_1_new['yday'].values, train_1_new['altitude'].values, s=2, alpha=0.2)
    plt.title('这张图展示了一年到头来每天太阳仰角的范围变化情况')

    fig = plt.figure(figsize=(20, 10))
    plt.scatter(train_1_new['yday'].values, train_1_new['distance'].values, s=2, alpha=0.2)
    plt.title('这张图展示了一年到头来每天距离太阳距离的变化情况')

    fig = plt.figure(figsize=(20, 10))
    plt.scatter(train_1_new['yday'].values, train_1_new['diameter'].values, s=2, alpha=0.2)
    plt.title('这张图展示了一年到头来每天看到的太阳直径的变化情况')

    fig = plt.figure(figsize=(20, 10))
    plt.scatter(train_1_new['distance'].values, train_1_new['diameter'].values, s=2, alpha=0.2)
    plt.title('这张图证明了距离太阳的距离和看到的太阳的直径是严格的函数关系，而且是线性关系')

    plt.show()


def contouring(series_1, series_2, series_y, grid_num, scope):
    feature_1_min = np.floor(series_1.min())
    feature_1_max = np.ceil(series_1.max())

    feature_2_min = np.floor(series_2.min())
    feature_2_max = np.ceil(series_2.max())

    the_xx, the_yy = np.meshgrid(np.linspace(feature_1_min, feature_1_max, grid_num),
                                 np.linspace(feature_2_min, feature_2_max, grid_num))

    z = []
    for j in range(grid_num):
        for i in range(grid_num):
            x = the_xx[0, :][i]
            y = the_yy[:, 0][j]

            condition_1 = (np.abs(series_1.values - x) <= scope * (feature_1_max - feature_1_min))
            condition_2 = (np.abs(series_2.values - y) <= scope * (feature_2_max - feature_2_min))

            tmp = series_y.values[condition_1 & condition_2]
            if len(tmp) == 0:
                result = np.nan
            else:
                result = np.nanmean(tmp)
            z.append(result)

    the_zz = np.array(z).reshape(the_xx.shape)

    return the_xx, the_yy, the_zz


def roll_contour(figure, fit_start, fit_end, grid_num):
    """
    获取rolling之后得到的等高线
    """
    the_vertices_all = np.array([]).reshape(-1, 2)
    for i in range(fit_start, fit_end):
        vertices = figure.collections[i].get_paths()[0].vertices
        vertices[:, 1] = vertices[:, 1] - np.mean(vertices[:, 1])
        the_vertices_all = np.concatenate([the_vertices_all, vertices], axis=0)

    vertices_all_df = pd.DataFrame(the_vertices_all, columns=['x', 'y'])
    vertices_all_df.sort_values(by='x', inplace=True)

    gap = np.int(len(vertices_all_df) / grid_num)
    point_num = int(np.ceil(len(vertices_all_df) / gap))

    agg_index = reduce(lambda x, y: x + y, [[i] * gap for i in range(point_num)])
    vertices_all_df['agg_index'] = agg_index[:len(vertices_all_df)]
    the_vertices_all_df_agg = vertices_all_df.rolling(window=100, min_periods=0).mean()

    return the_vertices_all, the_vertices_all_df_agg


def get_contour(data_train, data_test, grid_num, feature_1, feature_2, scope, line_num=20, fit_start=2, fit_end=13):
    xx, yy, zz = contouring(data_train[feature_1], data_train[feature_2], data_train['实际功率'], grid_num, scope)

    # 下面三行只是为了获取等高线，不画图
    plt.figure()
    fig_object = plt.contourf(xx, yy, zz, line_num, alpha=0.75, cmap=plt.cm.hot)
    plt.close()

    vertices_all, vertices_all_df_agg = roll_contour(fig_object, fit_start, fit_end, grid_num)

    # 下面对等高线进行拟合
    weights = np.polyfit(vertices_all_df_agg['x'].values, vertices_all_df_agg['y'].values, 10)
    y_predict_poly = np.polyval(weights, vertices_all_df_agg['x'].values)

    # 下面得到新的特征
    y_residual_train_ = data_train[feature_2] - np.polyval(weights, data_train[feature_1].values)
    y_residual_test_ = data_test[feature_2] - np.polyval(weights, data_test[feature_1].values)
    the_corr = pd.Series(y_residual_train_).corr(data_train['实际功率'])

    # 下面开始可视化
    fig_in = plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(vertices_all_df_agg['x'].values, vertices_all_df_agg['y'].values, color='blue')
    plt.plot(vertices_all_df_agg['x'].values, y_predict_poly, color='green')
    plt.scatter(vertices_all[:, 0], vertices_all[:, 1], s=5, alpha=1)
    plt.contourf(xx, yy, zz, line_num, alpha=0.75, cmap=plt.cm.hot)
    cc = plt.contour(xx, yy, zz, line_num, colors='black')
    plt.clabel(cc, inline=True, fontsize=10)
    plt.xlabel(feature_1, fontsize=20)
    plt.ylabel(feature_2, fontsize=20)

    plt.subplot(1, 2, 2)
    xx_res, yy_res, zz_res = contouring(data_train[feature_1], pd.Series(y_residual_train_),
                                        data_train['实际功率'], grid_num, scope)
    plt.contourf(xx_res, yy_res, zz_res, line_num, alpha=0.75, cmap=plt.cm.hot)
    plt.xlabel(feature_1, fontsize=20)
    plt.ylabel('%s_rectified' % feature_2, fontsize=20)

    plt.show()
    # plt.close()
    # print('%s:%s' % (feature_1, train_1_new[feature_1].corr(train_1_new['实际功率'])))
    # print('%s:%s' % (feature_2, train_1_new[feature_2].corr(train_1_new['实际功率'])))

    return fig_in, y_residual_train_, y_residual_test_, the_corr


def feature_cross(data_train, data_test):
    train_feature = pd.DataFrame()
    test_feature = pd.DataFrame()

    columns = ['辐照度', '风速', '风向', '温度', '压强', '湿度', 'azimuth',
               'altitude', 'distance', 'hours_float', 'mday', 'yday']
    for i in range(len(columns)):
        print('begin: %s' % str(i))
        feature_1 = columns[i]
        for j in range(i + 1, len(columns)):
            feature_2 = columns[j]

            _, y_residual_train_1, y_residual_test_1, corr_1 = get_contour(data_train, data_test, 100,
                                                                           feature_1, feature_2, 0.1)
            _, y_residual_train_2, y_residual_test_2, corr_2 = get_contour(data_train, data_test, 100,
                                                                           feature_2, feature_1, 0.1)

            col_name_1 = feature_1 + '_' + feature_2
            col_name_2 = feature_2 + '_' + feature_1

            train_feature[col_name_1] = y_residual_train_1
            train_feature[col_name_2] = y_residual_train_2
            test_feature[col_name_1] = y_residual_test_1
            test_feature[col_name_2] = y_residual_test_2

        print('end: %s' % str(i))

    data_train_new = pd.concat([data_train, train_feature], axis=1)
    data_test_new = pd.concat([data_test, test_feature], axis=1)

    return data_train_new, data_test_new


if __name__ == '__main__':
    train_1_new = pd.read_csv('./data_new/train_1_new.csv')
    train_2_new = pd.read_csv('./data_new/train_2_new.csv')
    train_3_new = pd.read_csv('./data_new/train_3_new.csv')
    train_4_new = pd.read_csv('./data_new/train_4_new.csv')

    test_1_new = pd.read_csv('./data_new/test_1_new.csv')
    test_2_new = pd.read_csv('./data_new/test_2_new.csv')
    test_3_new = pd.read_csv('./data_new/test_3_new.csv')
    test_4_new = pd.read_csv('./data_new/test_4_new.csv')

    get_plot(train_1_new)
    visualize_solar_data()

    x_0, y_residual_train_s, _, _ = get_contour(train_1_new, test_1_new, 100, '温度', '湿度', 0.1)
    x_1, _, _, _ = get_contour(train_1_new, test_1_new, 100, '压强', '温度', 0.2)
    x_2, _, _, _ = get_contour(train_1_new, test_1_new, 100, '湿度', '辐照度', 0.1, fit_end=15)
    x_3, _, _, _ = get_contour(train_1_new, test_1_new, 100, '温度', '辐照度', 0.1, fit_start=3, fit_end=15)
    x_4, _, _, _ = get_contour(train_1_new, test_1_new, 100, '风速', '辐照度', 0.1)
    x_5, _, _, _ = get_contour(train_1_new, test_1_new, 100, '风向', '辐照度', 0.1)
    x_6, _, _, _ = get_contour(train_1_new, test_1_new, 100, '压强', '辐照度', 0.1)
    x_7, _, _, _ = get_contour(train_1_new, test_1_new, 100, 'month', '辐照度', 0.1)
    x_8, _, _, _ = get_contour(train_1_new, test_1_new, 100, 'yday', '辐照度', 0.1)
    x_9, _, _, _ = get_contour(train_1_new, test_1_new, 100, 'mday', '辐照度', 0.1, line_num=50)
    x_10, _, _, _ = get_contour(train_1_new, test_1_new, 100, 'azimuth', '辐照度', 0.1)  # a good example
    x_11, _, _, _ = get_contour(train_1_new, test_1_new, 100, 'altitude', '辐照度', 0.1, fit_end=8)
    x_12, _, _, _ = get_contour(train_1_new, test_1_new, 100, 'azimuth', 'altitude', 0.1)
    x_13, _, _, _ = get_contour(train_1_new, test_1_new, 100, 'distance', '辐照度', 0.1)
    x_14, _, _, _ = get_contour(train_1_new, test_1_new, 100, '温度', 'altitude', 0.1)

    fig = plt.figure(figsize=(20, 10))
    plt.scatter(train_1_new['yday'], train_1_new['辐照度'], s=2, alpha=0.2)

    fig = plt.figure(figsize=(20, 10))
    plt.scatter(train_1_new['yday'], train_1_new['辐照度'], s=2, alpha=0.2)

    fig = plt.figure(figsize=(20, 10))
    plt.scatter(train_1_new['湿度'], train_1_new['辐照度'], s=2, alpha=0.2)
    train_1_new['hours_float'].corr(train_1_new['azimuth'])

    train_1_cross, test_1_cross = feature_cross(train_1_new, test_1_new)
    train_2_cross, test_2_cross = feature_cross(train_2_new, test_2_new)
    train_3_cross, test_3_cross = feature_cross(train_3_new, test_3_new)
    train_4_cross, test_4_cross = feature_cross(train_4_new, test_4_new)

    train_1_cross.to_csv('./data_cross/train_1_cross.csv', index=False)
    train_2_cross.to_csv('./data_cross/train_2_cross.csv', index=False)
    train_3_cross.to_csv('./data_cross/train_3_cross.csv', index=False)
    train_4_cross.to_csv('./data_cross/train_4_cross.csv', index=False)

    test_1_cross.to_csv('./data_cross/test_1_cross.csv', index=False)
    test_2_cross.to_csv('./data_cross/test_2_cross.csv', index=False)
    test_3_cross.to_csv('./data_cross/test_3_cross.csv', index=False)
    test_4_cross.to_csv('./data_cross/test_4_cross.csv', index=False)

    cross_col = [x for x in train_1_cross.columns.drop('hours_float') if len(x.split('_')) > 1]


data_train = train_1_cross[train_1_cross['实际功率'] > 0.03 * train_1_cross['实际功率'].max()]

import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as stats

data = data_train




lgb_eval = lgb.Dataset(X_test, y_test, reference=train_matrix)
model = lgb.train(params,
                  train_matrix,
                  num_round,
                  valid_sets=lgb_eval,
                  early_stopping_rounds=early_stopping_rounds,
                  verbose_eval=1000)

y_pred = model.predict(new_test.fillna(-999))


def naive_lgb(data_x, data_y):
    params = {'learning_rate': 0.084, 'max_depth': -1, 'metric': 'mae',
              'min_data': 6, 'min_child_weight': 0.001, 'num_leaves': 100,
              'objective': 'regression', 'lambda_l2': 1.1, 'nthread': 4,
              'early_stopping_rounds': 100, 'verbose_eval': 50,
              'num_boost_round': 1000, 'sub_feature': 0.9}
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=678)

    train_data = lgb.Dataset(x_train, y_train)
    valid_data = lgb.Dataset(x_test, y_test)

    lgb_model = lgb.train(params, train_set=train_data, valid_sets=valid_data)
    y_predict_test = lgb_model.predict(x_test)

    return metrics.mean_absolute_error(y_test, y_predict_test)


def naive_predict(data, degree=1, x=['辐照度'], y='实发辐照度'):
    """
    x里必须要有’辐照度‘
    """
    polynomial = PolynomialFeatures(degree=degree)  # 二次多项式
    x_transformed = polynomial.fit_transform(data[x])

    linear_reg_0 = LinearRegression()
    linear_reg_0.fit(data[['辐照度']], data['实际功率'])
    y_predict_0 = linear_reg_0.predict(data[['辐照度']])

    linear_reg_1 = LinearRegression()
    linear_reg_1.fit(x_transformed, data[y])
    agg_feature = linear_reg_1.predict(x_transformed)

    linear_reg_2 = LinearRegression()
    linear_reg_2.fit(pd.DataFrame({'agg_feature': agg_feature}), data['实际功率'])
    y_predict_2 = linear_reg_2.predict(pd.DataFrame({'agg_feature': agg_feature}))

    mae_0 = metrics.mean_absolute_error(y_predict_0, data['实际功率'])
    corr = stats.pearsonr(data['辐照度'].values, agg_feature)
    mae_2 = metrics.mean_absolute_error(y_predict_2, data['实际功率'])

    plt.scatter(data['辐照度'].values, data[y], s=2, alpha=0.3)
    plt.scatter(data['辐照度'].values, agg_feature, s=2, alpha=0.3)

    print('辐照度 预测 “实际功率”的mae', mae_0)
    print('对“实发辐照度”的预测值与“实发辐照度”的相关性', corr)
    print('对“实发辐照度”的预测值 预测 “实际功率”的mae', mae_2)

    return mae_2


temp = list(map(lambda x: naive_predict(data_train, degree=x), list(range(1, 20))))
plt.plot(range(1, len(temp) + 1), temp)

#
data_x, data_y = data_train.drop(['时间', '实发辐照度', '实际功率', 'diameter'], axis=1), data_train['实发辐照度']


data_train_2 = data_train.copy()
data_train_2['实发辐照度的预测值'] = agg_feature
data_train_2['实发辐照度的预测值_1'] = agg_feature
data_train_2['实发辐照度的预测值_2'] = agg_feature
data_train_2['实发辐照度的预测值_3'] = agg_feature
data_train_2['实发辐照度的预测值_4'] = agg_feature
data_train_2['实发辐照度的预测值_5'] = agg_feature
data_train_2['实发辐照度的预测值_6'] = agg_feature

data_x, data_y = data_train_2.drop(['时间', '实发辐照度', '实际功率', 'diameter'], axis=1), data_train_2['实际功率']

print(naive_lgb(data_x, data_y))















