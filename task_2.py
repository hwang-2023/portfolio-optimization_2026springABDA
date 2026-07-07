# task_2.py
import task_2_func
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# 加载估计出的 mu 和 sigma
loaded = np.load('mu_n_sigma.npz')
mu = loaded['mu']
sigma = loaded['sigma']
loaded.close()

# 定义 gamma 的取值列表（用于 A2 模型）
gamma_list = [np.linspace(0.0001, 0.001, 10),
              np.linspace(0.0015, 0.01, 18),
              np.linspace(0.015, 0.1, 18),
              np.linspace(0.15, 1.0, 18),
              np.linspace(1.5, 10.0, 18),
              np.linspace(20.0, 50.0, 4)]
gamma_list = np.concatenate(gamma_list)

# 存储 A2 模型的结果：风险、收益、权重
risk_a2 = []
return_a2 = []
weight_a2 = []
time_a2_list=[]
# 对每个 gamma 求解 A2 模型（均值-方差优化，带风险厌恶系数）
for gamma in gamma_list:
    start = time.perf_counter()
    x = task_2_func.solve_a2_by_cvx(gamma, sigma, mu)
    end = time.perf_counter()
    time_a2_list.append(end - start)
    port_return = mu @ x
    port_risk = np.sqrt(x.T @ sigma @ x)
    risk_a2.append(port_risk)
    return_a2.append(port_return)
    weight_a2.append(x)

# 定义 sigma_max 的取值列表（用于 B 模型，风险上限）
sigma_max_list = [np.linspace(0.0001, 0.001, 19),
                  np.linspace(0.0015, 0.01, 18),
                  np.linspace(0.012, 0.1, 45)]
sigma_max_list = np.concatenate(sigma_max_list)

# 存储 B 模型的结果
risk_b = []
return_b = []
time_b_list=[]
# 对每个 sigma_max 求解 B 模型（在给定风险上限下最大化收益）
for sigma_max in sigma_max_list:
    start = time.perf_counter()
    x = task_2_func.solve_b_by_cvx(sigma, mu, sigma_max)
    end = time.perf_counter()
    time_b_list.append(end - start)
    if x is None:   # 求解失败则跳过
        print(sigma_max)
        continue
    port_return = mu @ x
    port_risk = np.sqrt(x.T @ sigma @ x)
    risk_b.append(port_risk)
    return_b.append(port_return)


print("\ntime_a2")
print(time_a2_list)
print("\ntime_b")
print(time_b_list)

# 绘制有效前沿（风险-收益图）
plt.figure(1)
plt.plot(risk_a2, return_a2, 'o-', label='Model A2')
plt.plot(risk_b, return_b, 's-', label='Model B')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.grid(True)
plt.legend()

# 绘制资产占比堆叠图（针对 A2 模型）
weights = np.array(weight_a2)
risks = np.array(risk_a2)

# 按照风险从小到大排序
sort_idx = np.argsort(risks)
risks_sorted = risks[sort_idx]
weights_sorted = weights[sort_idx, :]

# 只保留最大权重超过1%的资产，其余归为“others”
max_weights = np.max(weights_sorted, axis=0)
keep_assets = (max_weights >= 0.01)

# 读取资产名称列表
asset_names = pd.read_csv('cleaned_assets.csv', header=None)[0].tolist()
kept_asset_names = [asset_names[i] for i, ifkeep in enumerate(keep_assets) if ifkeep]
labels = kept_asset_names + ['others']

# 构建用于堆叠图的权重矩阵
weights_main = weights_sorted[:, keep_assets]
weights_others = weights_sorted[:, ~keep_assets].sum(axis=1)
weights_plot = np.column_stack([weights_main, weights_others])

# 绘制堆叠面积图
plt.figure(2)
plt.stackplot(risks_sorted, weights_plot.T, labels=labels, alpha=0.8)
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Asset Weight')
plt.title('Portfolio Weights')
plt.legend()
plt.grid(linestyle='--', alpha=0.5)

plt.show()