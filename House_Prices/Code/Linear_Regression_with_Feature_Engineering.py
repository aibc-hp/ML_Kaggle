# -*- coding: utf-8 -*-
# @Time    : 2024/4/24 17:02
# @Author  : aibc-hp
# @File    : Linear_Regression_with_Feature_Engineering.py
# @Project : House_Prices
# @Software: PyCharm

"""
可视化目标变量的分布，明显的呈现右偏分布；
通过目标变量的统计信息发现，最大值与均值、75% 点位值等差值较大，可能存在异常值；
分离类别特征和数值特征，方便后续处理；
可视化各数值特征与目标变量的关系，查看数值特征通常采用散点图；可以发现 TotalBsmtSF、1stFlrSF、GrLivArea 等特征与 SalePrice 有较为明显的线性关系；
计算各数值特征与目标变量的相关性，并筛选出相关系数大于等于 0.5 的数值特征；
可视化各类别特征的分布情况，查看类别特征通常采用箱线图；可以发现 Neighborhood、SaleType 等特征的分布差异是较为明显的；
根据观察，筛选出了 MSZoning、Neighborhood、Condition1、Condition2、HouseStyle、RoofMatl、Exterior1st、Exterior2nd、ExterQual、BsmtQual、PoolQC、MiscFeature、SaleType、SaleCondition 类别特征

对筛选出的数值特征做异常值处理；
对类别特征和数值特征做缺失值处理；
对类别特征做数据转换；

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder


class Visualization(object):
    def __init__(self, train_pth: str, test_pth: str, gt_pth: str) -> None:
        self.df_train = pd.read_csv(train_pth)
        self.df_test = pd.read_csv(test_pth)
        self.df_gt = pd.read_csv(gt_pth)

    def visualize_target(self) -> None:
        """
        sns.displot() 函数是一个多功能的图表生成函数，可根据数据和参数生成直方图、核密度估计图（KDE）或是两者的结合等；
        传入的数据为 pd.DataFrame 或 pd.Series，当 kde=True 时可同时绘制直方图和核密度估计图，参数 kind 有 hist、kde、ecdf 几种选择，默认为 hist；
        kind='hist' 只绘制直方图；kind='kde' 只绘制核密度估计图；kind='ecdf' 绘制经验累积分布图；
        rug=True 表示在 x 轴上绘制一个 rug plot（一种显示数据点位置的线条图），默认为 False；
        横轴是 SalePrice，纵轴默认是 Count，可以通过 stat='probability'、stat='density' 来切换纵轴变量；
        """
        # 可以通过 describe() 函数查看 Series 数据的统计信息
        print(self.df_train['SalePrice'].describe())

        # 可以通过计算偏度来检验数据的分布
        skewness = self.df_train['SalePrice'].skew()
        if skewness > 0:
            print('Data presents a right skewed distribution.')
        elif skewness < 0:
            print('Data presents a left skewed distribution.')
        else:
            print('Data presents a normal distribution.')

        # 可以通过 Shapiro-Wilk 检验来验证数据是否符合正态分布，如果 p-value 值高于显著性水平（如 0.05），则不能拒绝数据服从正态分布的假设
        shapiro_test = stats.shapiro(self.df_train['SalePrice'])
        if shapiro_test[1] > 0.05:
            print('Data presents a normal distribution.')
        else:
            print('Data presents a skewed distribution.')

        # 可以通过联合绘制直方图和核密度估计图来直观地查看数据分布的形状
        sns.displot(self.df_train['SalePrice'], kde=True)
        plt.title('Data distribution of target variable')

        plt.show()

    def visualize_num_features_and_target(self) -> None:
        df_train = self.df_train.copy()
        df_test = self.df_test.copy()

        # 删掉 Id 列
        df_train.drop('Id', axis=1, inplace=True)  # (1460, 80)
        df_test.drop('Id', axis=1, inplace=True)  # (1459, 79)

        # 分离类别特征和数值特征
        cate_features_lst = df_test.select_dtypes(include=['object']).columns.tolist()  # 43
        num_features_lst = df_test.select_dtypes(include=['int64', 'float64']).columns.tolist()  # 36

        # 绘制散点图
        plt.figure(figsize=(16, 20), constrained_layout=True)
        # plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 手动调整子图之间的间距
        # plt.tight_layout()  # 自动调整子图布局
        for i, feature in enumerate(num_features_lst):
            plt.subplot(9, 4, i+1)
            sns.scatterplot(data=df_train, x=feature, y='SalePrice', )  # 绘制散点图，data 为 pd.DataFrame，x、y 为列名
            plt.xlabel(feature)
            plt.ylabel('SalePrice')
        plt.suptitle('The relationship between various numerical features and the target variable')  # 给整个图形添加标题

        plt.show()

    def visualize_cate_features(self) -> None:
        df_train = self.df_train.copy()

        # 删掉 Id 列
        df_train.drop('Id', axis=1, inplace=True)  # (1460, 80)

        # 分离类别特征和数值特征
        cate_features_lst = df_train.select_dtypes(include=['object']).columns.tolist()  # 43
        num_features_lst = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()  # 37

        lst1 = cate_features_lst[:9]
        lst2 = cate_features_lst[9:18]
        lst3 = cate_features_lst[18:27]
        lst4 = cate_features_lst[27:36]
        lst5 = cate_features_lst[36:]

        # 绘制箱线图
        plt.figure(figsize=(16, 20), constrained_layout=True)
        for i, feature in enumerate(lst5):
            plt.subplot(3, 3, i + 1)
            sns.boxplot(data=df_train, x=feature, y='SalePrice', )  # 绘制箱线图，data 为 pd.DataFrame，x、y 为列名
            plt.xlabel(feature)
            plt.ylabel('SalePrice')
            plt.xticks(rotation=45)
        plt.suptitle('The distribution of 37-43 categorical features')  # 给整个图形添加标题

        plt.show()

    def visualize_corr_of_num_features(self) -> None:
        df_train = self.df_train.copy()

        # 删掉 Id 列
        df_train.drop('Id', axis=1, inplace=True)  # (1460, 80)

        # 分离类别特征和数值特征
        cate_features = df_train.select_dtypes(include=['object'])  # 43
        num_features = df_train.select_dtypes(include=['int64', 'float64'])  # 37
        train_num_features = num_features.drop('SalePrice', axis=1)  # 36

        # 绘制所有数值特征的相关系数热力图
        corrs = train_num_features.corr()  # 计算所有数值特征之间的相关性，得到 pd.DataFrame 结构的相关系数矩阵
        sns.heatmap(corrs, xticklabels=train_num_features.columns, yticklabels=train_num_features.columns)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title('The correlation between all numerical features')

        plt.show()

        # 绘制相关性前十的数值特征的相关系数热力图
        corr_dct = {}
        for col in num_features.columns:
            corr = num_features[col].corr(num_features['SalePrice'])
            corr_dct[col] = corr

        sorted_dct = dict(sorted(corr_dct.items(), key=lambda item: item[1], reverse=True))
        top_ten_cols = list(sorted_dct.keys())[:min(11, len(sorted_dct))]

        corrs_10 = num_features[top_ten_cols].corr()
        sns.heatmap(corrs_10, annot=True, fmt='.2f', xticklabels=top_ten_cols, yticklabels=top_ten_cols)

        plt.title('The correlation between top-10 numerical features')

        plt.show()


class FeatureEngineering(object):
    def __init__(self, train_pth: str, test_pth: str, gt_pth: str) -> None:
        self.df_train = pd.read_csv(train_pth)
        self.df_test = pd.read_csv(test_pth)
        self.df_gt = pd.read_csv(gt_pth)

    def get_features(self) -> (list, list):
        df_train = self.df_train.copy()
        df_train.drop('Id', axis=1, inplace=True)  # Id 列没有用，直接删掉

        # 分离类别特征和数值特征，后续处理会更方便
        # 可以通过检查数据类型（dtype）来区分类别特征和数值特征；通常，类别特征是 object 类型（字符串），而数值特征可以是 int64、float64 等类型
        # 某些情况下类别特征可能被错误地标记为 int64 类型，尤其是当类别是有序的或者是有限的无序集合时。在这种情况下，可能需要额外的逻辑来识别这些特征，例如检查数据的唯一值的数量，如果唯一值的数量较少，可能表明这是一个类别特征
        train_cate_features = df_train.select_dtypes(include=['object'])  # 43
        train_num_features = df_train.select_dtypes(include=['int64', 'float64'])  # 37
        # print(train_cate_features.columns.tolist())  # 查看列名

        # 计算 train_num_features 中各数值特征与目标变量之间的相关性
        train_num_features.drop('SalePrice', axis=1, inplace=True)  # 36
        corr_dct = {}
        for col in train_num_features.columns:
            corr = df_train[col].corr(df_train['SalePrice'])  # 计算 pd.DataFrame 中两个特征之间的相关性
            corr_dct[col] = corr

        filtered_corr_dct = {k: v for k, v in corr_dct.items() if v >= 0.5}  # 筛选出相关系数大于等于 0.5 的数值特征
        train_num_features_select = list(filtered_corr_dct.keys())

        train_cate_features_select = ['MSZoning', 'Neighborhood', 'Condition1', 'Condition2', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'BsmtQual', 'PoolQC', 'MiscFeature', 'SaleType', 'SaleCondition']

        return train_num_features_select, train_cate_features_select

    def outlier_process(self) -> pd.DataFrame:
        """
        数值特征选择了 'TotalBsmtSF'、'1stFlrSF'、'GrLivArea'、'GarageArea'
        """
        num_feat_select = ['TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea']
        df_train = self.df_train[num_feat_select]

        # 处理 'TotalBsmtSF' 特征的异常值
        sorted_feat = sorted(df_train['TotalBsmtSF'])
        df_train['TotalBsmtSF'].iloc[df_train[df_train['TotalBsmtSF'] == sorted_feat[-1]].index] = sorted_feat[-2]

        # 处理 '1stFlrSF' 特征的异常值
        sorted_feat = sorted(df_train['1stFlrSF'])
        df_train['1stFlrSF'].iloc[df_train[df_train['1stFlrSF'] == sorted_feat[-1]].index] = sorted_feat[-2]

        # 处理 'GrLivArea' 特征的异常值
        sorted_feat = sorted(df_train['GrLivArea'])
        df_train['GrLivArea'].iloc[df_train[df_train['GrLivArea'] == sorted_feat[-2]].index] = sorted_feat[-3]
        df_train['GrLivArea'].iloc[df_train[df_train['GrLivArea'] == sorted_feat[-1]].index] = sorted_feat[-3]

        return df_train

    def missing_values_process(self, cate_feat_select: list, df_train_num_feat_select: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        df_train = self.df_train.copy()
        df_test = self.df_test.copy()

        df_train = df_train[cate_feat_select]
        df_train = pd.concat((df_train, df_train_num_feat_select), axis=1)
        cate_feat_select.extend(['TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea'])
        df_test = df_test[cate_feat_select]

        print('The shape of training data:', df_train.shape)  # (1460, 18)
        train_missing = df_train.isnull().sum()  # 获取训练数据中各特征的缺失值个数
        train_missing = train_missing.drop(train_missing[train_missing == 0].index).sort_values(ascending=False)  # 3

        print('The shape of training data:', df_test.shape)  # (1459, 18)
        test_missing = df_test.isnull().sum()  # 获取测试数据中各特征的缺失值个数
        test_missing = test_missing.drop(test_missing[test_missing == 0].index).sort_values(ascending=False)  # 9

        # # 类别特征；根据特征说明文档，以下特征中的数据缺失代表没有，因此可以用 None 填充
        # none_lst = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'MasVnrType']
        # for col in none_lst:
        #     df_train[col] = df_train[col].fillna('None')
        #     df_test[col] = df_test[col].fillna('None')
        #
        # # 类别特征；根据特征说明文档，以下特征中的数据缺失不代表没有，而是丢失了，因此可以用出现次数最多的值填充
        # most_lst = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'SaleType', 'KitchenQual', 'Electrical']
        # for col in most_lst:
        #     # mode() 函数返回出现频率最高的值，如果有多个具有相同最高频率的值，则会全部返回
        #     df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
        #     df_test[col] = df_test[col].fillna(df_train[col].mode()[0])
        #
        # # 类别特征；根据特征说明文档，'Functional' 特征中的数据缺失直接填入 'Typ'
        # df_train['Functional'] = df_train['Functional'].fillna('Typ')
        # df_test['Functional'] = df_test['Functional'].fillna('Typ')

        # 类别特征；根据特征说明文档，以下特征中的数据缺失代表没有，因此可以用 None 填充
        none_lst = ['PoolQC', 'MiscFeature', 'BsmtQual']
        for col in none_lst:
            df_train[col] = df_train[col].fillna('None')
            df_test[col] = df_test[col].fillna('None')

        # 类别特征；根据特征说明文档，以下特征中的数据缺失不代表没有，而是丢失了，因此可以用出现次数最多的值填充
        most_lst = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'SaleType']
        for col in most_lst:
            # mode() 函数返回出现频率最高的值，如果有多个具有相同最高频率的值，则会全部返回
            df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
            df_test[col] = df_test[col].fillna(df_train[col].mode()[0])

        # 数值特征；根据特征说明文档，以下特征中的数据缺失可以用零来填充
        zero_lst = ['TotalBsmtSF', 'GarageArea']
        for col in zero_lst:
            df_train[col] = df_train[col].fillna(0)
            df_test[col] = df_test[col].fillna(0)

        # print(df_train.isnull().sum().any())  # 检查训练数据中是否还存在缺失值
        # print(df_test.isnull().sum().any())  # 检查测试数据中是否还存在缺失值

        return df_train, df_test

    def label_transform(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        df_train = df_train.select_dtypes(include=['object'])
        df_test = df_test.select_dtypes(include=['object'])

        for col in df_train.columns:
            encoder = LabelEncoder()
            value_unique_train = set(df_train[col].unique())
            value_unique_test = set(df_test[col].unique())
            value_lst = list(value_unique_train | value_unique_test)
            encoder.fit(value_lst)

            df_train[col] = encoder.transform(df_train[col])
            df_test[col] = encoder.transform(df_test[col])

        return df_train, df_test


def data_conversion(pth: str) -> (np.ndarray, np.ndarray):
    """
    Convert non-numerical data to numerical data.
    :param pth: The storage address of the training set or the test set.
    :return: Feature array and target array.
    """
    def na_to_0(data_series: pd.Series) -> None:
        # 使用 fillna() 方法将 DataFrame 中的 NaN 值替换为指定的值；inplace=True 表示在原 DataFrame 上进行替换操作，而不是返回一个新的 DataFrame
        data_series.fillna(0, inplace=True)

    def unique_map(data_series: pd.Series) -> pd.Series:
        # 使用 unique() 方法获取列中的唯一值，再使用 map() 方法根据映射字典将列中的值转换为数值型数据
        data_series = data_series.map({value: index for index, value in enumerate(data_series.unique())})
        return data_series

    df = pd.read_csv(pth)
    df.drop('Id', axis=1, inplace=True)  # Id 列没有用，直接删掉

    # 分离类别特征和数值特征，后续处理会更方便
    # 可以通过检查数据类型（dtype）来区分类别特征和数值特征；通常，类别特征是 object 类型（字符串），而数值特征可以是 int64、float64 等类型
    # 某些情况下类别特征可能被错误地标记为 int64 类型，尤其是当类别是有序的或者是有限的无序集合时。在这种情况下，可能需要额外的逻辑来识别这些特征，例如检查数据的唯一值的数量，如果唯一值的数量较少，可能表明这是一个类别特征
    cate_features = df.select_dtypes(include=['object'])  # 43
    num_features = df.select_dtypes(include=['int64', 'float64'])  # train: 37, test: 36
    # print(cate_features.columns.tolist())  # 查看列名

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


def rmse(pred_values: np.ndarray, pth: str) -> float:
    """
    Take the logarithm of the predicted and observed values, and then calculate the root mean square error between them.
    This is the evaluation indicator for predicting housing prices.
    :param pred_values: The predict values.
    :param pth: The storage address of the ground-truth values of the test set.
    :return: Evaluation indicator result.
    """
    df = pd.read_csv(pth)
    gt_target = df['SalePrice'].values

    errors = gt_target - pred_values

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

    df.to_csv(r'D:\Kaggle\House_Prices\Submission_File\lr_engin.csv', index=False)


if __name__ == '__main__':
    color = sns.color_palette(palette='pastel')
    sns.set_style('darkgrid')

    train_path = r'D:\Kaggle\House_Prices\Dataset\train.csv'
    test_path = r'D:\Kaggle\House_Prices\Dataset\test.csv'
    gt_path = r'D:\Kaggle\House_Prices\Dataset\sample_submission.csv'

    train_features, train_target = data_conversion(train_path)
    test_features = data_conversion(test_path)

    vis = Visualization(train_path, test_path, gt_path)
    # vis.visualize_target()
    # vis.visualize_num_features_and_target()
    # vis.visualize_cate_features()
    # vis.visualize_corr_of_num_features()

    feat_engin = FeatureEngineering(train_path, test_path, gt_path)
    num_features_select, cate_features_select = feat_engin.get_features()
    df_train_num_features = feat_engin.outlier_process()
    df_train_features, df_test_features = feat_engin.missing_values_process(cate_features_select, df_train_num_features)
    df_train_features, df_test_features = feat_engin.label_transform(df_train_features, df_test_features)

    pred_results = train(df_train_features.values, train_target, df_test_features.values)

    # residual_results = residual(pred_results, gt_path)

    log_rmse_result = rmse(pred_results, gt_path)
    print(log_rmse_result)

    # save_to_csv(pred_results)
