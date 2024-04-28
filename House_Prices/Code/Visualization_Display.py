# -*- coding: utf-8 -*-
# @Time    : 2024/4/24 17:39
# @Author  : aibc-hp
# @File    : Visualization_Display.py
# @Project : House_Prices
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors


def show_color_palette(cl: list):
    # 将 RGB 颜色转换为可以识别的十六进制颜色字符串
    hex_colors = [mcolors.to_hex(c) for c in cl]

    # 创建一个图形和子图
    fig, ax = plt.subplots(1, len(cl), figsize=(15, 5))

    # 遍历颜色列表并为每种颜色创建一个矩形
    for i, color in enumerate(cl):
        ax[i].set_title(hex_colors[i])
        ax[i].add_patch(plt.Rectangle((0, 0), 1, 1, color=mcolors.to_rgba(color)))
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0, 1)
        ax[i].axis('off')

    # 显示图形
    plt.show()


def show_set_style(mode):
    # 该函数用于设置 seaborn 图表的全局样式。这个样式会影响所有随后创建的图表的外观；有 darkgrid、whitegrid、dark、white、ticks 几种模式
    sns.set_style(mode)

    x = np.array(list(range(1, 101)))
    y = 0.2 * x + 1.0
    plt.scatter(x, y)

    plt.show()


# 该函数返回一个颜色列表，这个列表可以用于图表中的数据点
# palette 有 deep、muted、bright、pastel、dark、colorblind 几种模式
color_list = sns.color_palette(palette='colorblind')
show_color_palette(color_list)

show_set_style('ticks')





