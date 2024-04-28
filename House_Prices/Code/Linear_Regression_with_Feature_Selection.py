# -*- coding: utf-8 -*-
# @Time    : 2024/4/24 10:27
# @Author  : aibc-hp
# @File    : Linear_Regression_with_Feature_Selection.py
# @Project : House_Prices
# @Software: PyCharm

"""
1. 原始特征（包含 Id 列），做了缺失值填充和数据转换，预测结果为 Log_RMSE = 1.0081922030758321
2. 原始特征（删除 Id 列），做了缺失值填充和数据转换，对特征做了标准化处理，预测结果为 Log_RMSE = 15.161137394941743
3. 原始特征（删除 Id 列），做了缺失值填充和数据转换，对特征做了归一化处理，预测结果为 Log_RMSE = 26.46273351514826
4. 原始特征（删除 Id 列），做了缺失值填充和数据转换，对特征做了对数转换处理，预测结果为 Log_RMSE = 2.2342501921793887
4. 原始特征（删除 Id 列），做了缺失值填充和数据转换，使用方差选择法进行特征筛选，预测结果为 Log_RMSE = 0.09198898927581334
5. 原始特征（删除 Id 列），做了缺失值填充和数据转换，使用相关系数法进行特征筛选，预测结果为 Log_RMSE = 0.09083499335635332
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy.stats import pearsonr


def data_conversion(pth: str) -> (np.ndarray, np.ndarray):
    """
    Convert non-numerical data to numerical data.
    :param pth: The storage address of the training set or the test set.
    :return: Feature array and target array.
    """
    def na_to_0(data_frame: pd.DataFrame) -> None:
        # 使用 fillna() 方法将 DataFrame 中的 NaN 值替换为指定的值；inplace=True 表示在原 DataFrame 上进行替换操作，而不是返回一个新的 DataFrame
        data_frame.fillna(0, inplace=True)

    def unique_map(data_frame: pd.DataFrame) -> pd.DataFrame:
        # 使用 unique() 方法获取列中的唯一值，再使用 map() 方法根据映射字典将列中的值转换为数值型数据
        data_frame = data_frame.map({value: index for index, value in enumerate(data_frame.unique())})
        return data_frame

    df = pd.read_csv(pth)
    df.drop('Id', axis=1, inplace=True)  # Id 列没有用，直接删掉

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = unique_map(df[column])
        else:
            na_to_0(df[column])

    if 'train.csv' in pth:
        target = df.pop('SalePrice')  # pop() 方法会从原 DataFrame 中删除指定列，并返回该列作为一个 Series 对象
        df_arr = df.values  # 将 DataFrame 转换成 NumPy 数组
        target_arr = target.values  # 将 Series 转换成 NumPy 数组
        return df_arr, target_arr
    elif 'test.csv' in pth:
        df_arr = df.values
        return df_arr
    else:
        raise ValueError


def standarize(features: np.ndarray) -> np.ndarray:
    """
    Standarize the training features or the test features.
    :param features: The training features or the test features.
    :return: Standarized features.
    """
    for i in range(features.shape[1]):
        features[:, i] = (features[:, i] - np.mean(features[:, i])) / np.std(features[:, i])

    return features


def normalize(features: np.ndarray) -> np.ndarray:
    """
    Normalize the training features or the test features.
    :param features: The training features or the test features.
    :return: Normalized features.
    """
    for i in range(features.shape[1]):
        features[:, i] = (features[:, i] - np.min(features[:, i])) / (np.max(features[:, i]) - np.min(features[:, i]))

    return features


def log_transformation(features: np.ndarray) -> np.ndarray:
    """
    Perform logarithmic conversion on training or testing features.
    :param features: The training features or the test features.
    :return: Transformed features.
    """
    for i in range(features.shape[1]):
        features[:, i] = np.log(np.where(features[:, i] <= 0, 1e-8, features[:, i]))

    return features


def variance_selection(train_feat: np.ndarray, test_feat: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Using variance selection method for feature selection.
    :param train_feat: The training features.
    :param test_feat: The test features.
    :return: Selected features.
    """
    features_variance = []
    for i in range(train_feat.shape[1]):
        features_variance.append(np.var(train_feat[:, i]))

    # print(min(features_variance))  # 0.00068446237568024
    # print(max(features_variance))  # 0.11831691165738832

    # select_features_idx = [index for index, value in enumerate(features_variance) if value > 0.075]  # log_rmse: 0.2614667812072453
    # select_features_idx1 = [index for index, value in enumerate(features_variance) if value < 0.001]  # log_rmse: 0.09198898927581334
    # select_features_idx.extend(select_features_idx1)  # log_rmse: 0.2617521097489771

    select_features_idx = [index for index, value in enumerate(features_variance) if value < 0.001]

    train_feat = train_feat[:, select_features_idx]
    test_feat = test_feat[:, select_features_idx]

    return train_feat, test_feat


def correlation_coe(train_feat: np.ndarray, test_feat: np.ndarray, train_tar: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Using correlation coefficient method for feature selection.
    :param train_feat: The training features.
    :param test_feat: The test features.
    :param train_tar: The training target.
    :return: Selected features.
    """
    pearsonr_correlation_coe = []
    for i in range(train_feat.shape[1]):
        _, p = pearsonr(train_feat[:, i], train_tar)
        pearsonr_correlation_coe.append(p)

    # print(max(pearsonr_correlation_coe))  # 0.8536006226782246

    select_features_idx = [index for index, value in enumerate(pearsonr_correlation_coe) if value > 0.75]

    train_feat = train_feat[:, select_features_idx]
    test_feat = test_feat[:, select_features_idx]

    return train_feat, test_feat


def train(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Training linear regression model, and obtain predict values.
    :param X_train: The training features.
    :param y_train: The training target.
    :param X_test: The test features.
    :return: Predict values.
    """
    lr_reg = linear_model.LinearRegression()
    lr_reg.fit(X_train, y_train)
    predicts = lr_reg.predict(X_test)

    return predicts


def residual(pred_values: np.ndarray, pth: str) -> np.ndarray:
    """
    Calculate the residual between predict values and ground-truth values.
    :param pred_values: The predict values.
    :param pth: The storage address of the ground-truth values of the test set.
    :return: The residual values.
    """
    df = pd.read_csv(pth)
    gt_target = df['SalePrice'].values

    result = gt_target - pred_values

    return result


def log_rmse(pred_values: np.ndarray, pth: str) -> float:
    """
    Take the logarithm of the predicted and observed values, and then calculate the root mean square error between them.
    This is the evaluation indicator for predicting housing prices.
    :param pred_values: The predict values.
    :param pth: The storage address of the ground-truth values of the test set.
    :return: Evaluation indicator result.
    """
    df = pd.read_csv(pth)
    gt_target = df['SalePrice'].values

    pred_values = np.where(pred_values <= 1, 1, pred_values)
    gt_target = np.where(gt_target <= 1, 1, gt_target)

    log_pred_values = np.log(pred_values)
    log_gt_target = np.log(gt_target)

    errors = log_gt_target - log_pred_values

    rmse = np.sqrt(np.mean(errors ** 2))

    return rmse


def save_to_csv(pred_values: np.ndarray) -> None:
    """
    Saving the predict results to a new csv file.
    :param pred_values: Predict results.
    :return: None.
    """
    ids = list(range(1461, 2920))
    df = pd.DataFrame({'Id': ids})

    df['SalePrice'] = pred_values

    df.to_csv(r'D:\Kaggle\House_Prices\lr_corr.csv', index=False)


if __name__ == '__main__':
    train_path = r'D:\Kaggle\House_Prices\train.csv'
    test_path = r'D:\Kaggle\House_Prices\test.csv'
    gt_path = r'D:\Kaggle\House_Prices\sample_submission.csv'

    train_features, train_target = data_conversion(train_path)
    test_features = data_conversion(test_path)

    # Standarize
    train_features_std = standarize(train_features)
    test_features_std = standarize(test_features)

    # Normalize
    train_features_norm = normalize(train_features)
    test_features_norm = normalize(test_features)

    # Log transformation
    # train_features_log_trans = log_transformation(train_features)
    # test_features_log_trans = log_transformation(test_features)

    # Variance selection
    train_features_var, test_features_var = variance_selection(train_features, test_features)

    # Correlation coefficient
    train_features_corr, test_features_corr = correlation_coe(train_features, test_features, train_target)

    pred_results = train(train_features_corr, train_target, test_features_corr)

    # residual_results = residual(pred_results, gt_path)

    log_rmse_result = log_rmse(pred_results, gt_path)
    print(log_rmse_result)

    # save_to_csv(pred_results)
