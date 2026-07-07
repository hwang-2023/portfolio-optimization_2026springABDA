# task_1.py
import pandas as pd
import numpy as np
import task_1_func_1
import task_1_func_2

# 执行数据下载
task_1_func_1.data_download()

# 读取原始价格数据，第一列为索引（日期），解析日期格式
original_data = pd.read_csv('original_data.csv', index_col=0, parse_dates=True)
print("\noriginal_data: ", original_data.shape)

# 调用清洗函数，清洗原始数据并保存为 cleaned_data.csv
task_1_func_2.data_clean(original_data)

# 读取清洗后的价格数据
cleaned_data = pd.read_csv('cleaned_data.csv', index_col=0, parse_dates=True)
print("\ncleaned_data: ", cleaned_data.shape)

# 调用收益率清洗函数，计算并清洗收益率，保存为 cleaned_returns.csv
task_1_func_2.returns_clean(cleaned_data)

# 读取清洗后的收益率数据
cleaned_returns = pd.read_csv('cleaned_returns.csv', index_col=0, parse_dates=True)
print("\ncleaned_returns: ", cleaned_returns.shape)

# 存储资产名称（列名），用于后续 task_2 绘图
cleaned_assets = cleaned_returns.columns.tolist()
pd.Series(cleaned_assets).to_csv('cleaned_assets.csv', header=False, index=False)

# 定义用于参数估计的时间窗口：最后252个交易日
begin_index = len(cleaned_returns.index) - 252  # 起始行索引
end_index = len(cleaned_returns.index) - 1     # 结束行索引

# 估计收益率均值向量 mu 和协方差矩阵 sigma（已正则化）
mu, sigma = task_1_func_2.parameters_estimation(cleaned_returns, begin_index, end_index)

# 保存 mu 和 sigma 为 npz 文件
np.savez('mu_n_sigma.npz', mu=mu, sigma=sigma)

# 输出完成信息及估计结果的形状和部分值
print("\n完成数据的估计, 存储至 mu_n_sigma.npz!")
print(f"时间范围: {cleaned_returns.index[begin_index]} 到 {cleaned_returns.index[end_index]}")
print("估计的 mu 向量形状: ", mu.shape)
print("估计的 sigma 矩阵形状: ", sigma.shape)
print("mu 的前 4 个值: ", mu[:4])
print("sigma 的前 4x4 子矩阵:\n", sigma[:4, :4])