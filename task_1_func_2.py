# task_1_func_2.py
import pandas as pd
import numpy as np

# 清洗原始价格数据并保存为 cleaned_data.csv
# 输入: 原始价格DataFrame d (索引为日期，列为资产)
def data_clean(d):
    print(f"清洗前资产数: {d.shape[1]}")

    # 删除缺失率超过5%的资产
    missing_rate = d.isnull().sum() / len(d)
    d = d[missing_rate[missing_rate < 0.05].index]

    # 删除任何包含NaN的行（即删除存在缺失的交易日）
    d = d.dropna()

    # 向前填充后向后填充，处理退市/停牌导致的价格缺失
    d = d.ffill().bfill()

    # 注：价格已经是调整后收盘价（Adj Close）

    # 保存清洗后的价格数据
    d.to_csv('cleaned_data.csv')
    print(f"\n清洗后数据保存成功!")
    print(f"清洗后资产数: {d.shape[1]}")
    print(f"时间范围: {d.index[0]} 到 {d.index[-1]}")
    print(f"交易日数量: {d.shape[0]}")
    print(f"保存文件: cleaned_data.csv")

    return 0

# 计算收益率并清洗，保存为 cleaned_returns.csv
# 输入: 清洗后的价格DataFrame d_c
def returns_clean(d_c):
    # 计算简单收益率
    returns = d_c.pct_change()

    # 删除第一行（NaN）
    returns = returns.dropna()

    # 处理异常值：去除超过10个标准差的日收益率
    mean_returns = returns.mean()      # 每列均值
    std_returns = returns.std()        # 每列标准差
    # 保留满足 |r - mean| <= 10*std 的观测值（此处是对整个DataFrame逐元素筛选）
    returns = returns[(returns - mean_returns).abs() <= 10 * std_returns]

    # 保存清洗后的收益率数据
    returns.to_csv('cleaned_returns.csv')
    print(f"\n清洗后收益率数据保存成功!")
    print(f"清洗后资产数: {returns.shape[1]}")
    print(f"时间范围: {returns.index[0]} 到 {returns.index[-1]}")
    print(f"交易日数量: {returns.shape[0]}")
    print(f"保存文件: cleaned_returns.csv")

    return 0

# 估计收益率均值向量和协方差矩阵（添加正则化）
# 输入: 清洗后收益率DataFrame d, 起始行索引 begin_index, 结束行索引 end_index (包含)
# 输出: 均值向量 mu (numpy数组), 正则化协方差矩阵 sigma_reg
def parameters_estimation(d, begin_index, end_index):
    # 选取指定时间窗口的数据
    d = d.iloc[begin_index:end_index + 1]

    # 计算均值向量和协方差矩阵（默认无偏估计，分母为 N-1）
    mu = d.mean().values
    sigma = d.cov().values
    # 添加极小正则化项，确保矩阵正定
    sigma_reg = sigma + 1e-6 * np.eye(len(mu))

    return mu, sigma_reg