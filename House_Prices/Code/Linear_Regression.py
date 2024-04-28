# -*- coding: utf-8 -*-
# @Time    : 2024/4/23 15:10
# @Author  : aibc-hp
# @File    : Linear_Regression.py
# @Project : House_Prices
# @Software: PyCharm

import numpy as np
import pandas as pd
from sklearn import linear_model


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


def save_to_csv(pred_values: np.ndarray) -> None:
    """
    Saving the predict results to a new csv file.
    :param pred_values: Predict results.
    :return: None.
    """
    ids = list(range(1461, 2920))
    df = pd.DataFrame({'Id': ids})

    df['SalePrice'] = pred_values

    df.to_csv(r'D:\Kaggle\House_Prices\lr.csv', index=False)


if __name__ == '__main__':
    train_path = r'D:\Kaggle\House_Prices\train.csv'
    test_path = r'D:\Kaggle\House_Prices\test.csv'
    gt_path = r'D:\Kaggle\House_Prices\sample_submission.csv'

    train_features, train_target = data_conversion(train_path)
    test_features = data_conversion(test_path)

    pred_results = train(train_features, train_target, test_features)

    residual_results = residual(pred_results, gt_path)

    # save_to_csv(pred_results)
