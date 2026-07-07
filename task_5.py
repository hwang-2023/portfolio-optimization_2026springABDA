# task_5.py
import pandas as pd
import task_1_func_2
import task_2_func
import task_3_func
import task_5_func
import numpy as np
import matplotlib.pyplot as plt

# 回测参数
L = 252          # 估计窗口长度（交易日）
H = 20           # 调仓频率（每 H 个交易日调仓一次）

# 待测试的参数
gamma_list = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]   # QP 模型的风险厌恶系数
sigma_max_list = [0.005, 0.01, 0.02, 0.05]                # SOCP 模型的风险上限

# 读取清洗后的收益率和价格数据
returns = pd.read_csv('cleaned_returns.csv', index_col=0, parse_dates=True)
prices = pd.read_csv('cleaned_data.csv', index_col=0, parse_dates=True)

# 确保 prices 与 returns 的资产列一致，并按照 returns 的日期对齐
common_assets = returns.columns
prices = prices[common_assets]
prices = prices.loc[returns.index]

# 生成调仓日期列表：从第 L 个交易日开始，每隔 H 个交易日调仓一次
rebalance_date_list = returns.index[L:][::H]

# 存储各策略的权重序列
weight_ew_list = []                     # 等权重
weight_qp_dict = {gamma: [] for gamma in gamma_list}      # QP 模型，每个 gamma 对应一个权重列表
weight_socp_dict = {sigma_max: [] for sigma_max in sigma_max_list}  # SOCP 模型

n = returns.shape[1]   # 资产数量

# 滚动回测
for rebalance_date in rebalance_date_list[:-1]:
    pos = returns.index.get_loc(rebalance_date)
    # 使用过去 L 天的收益率估计 mu 和 sigma
    mu, sigma = task_1_func_2.parameters_estimation(
        d=returns, begin_index=pos - L, end_index=pos - 1)

    # 等权重
    weight_ew_list.append(np.ones(n) / n)

    # QP 模型：使用 ADMM 求解（rho=0.01 固定）
    for gamma in gamma_list:
        x, _, _, _, _ = task_3_func.solve_a2_by_admm(gamma=gamma, rho=0.01, sigma=sigma, mu=mu)
        weight_qp_dict[gamma].append(x)

    # SOCP 模型：使用 CVXPY 求解
    for sigma_max in sigma_max_list:
        x = task_2_func.solve_b_by_cvx(sigma, mu, sigma_max=sigma_max)
        if x is None:                     # 若不可行则使用等权重
            x = np.ones(n) / n
        weight_socp_dict[sigma_max].append(x)

# 计算回测的总自然天数（用于年化计算）
idx1 = returns.index.get_loc(rebalance_date_list[0])
idx2 = returns.index.get_loc(rebalance_date_list[-1])
date1 = returns.index[idx1]
date2 = returns.index[idx2]
natural_days = (date2 - date1).days

data = []   # 存储各策略的绩效指标

# 等权重策略
money_list, returns_list = task_5_func.get_returns_by_weight(prices, rebalance_date_list, weight_ew_list)
result = task_5_func.get_result(money_list, returns_list, weight_ew_list, natural_days, H)
data.append(result)

# QP 策略（各 gamma）
for gamma in gamma_list:
    money_list, returns_list = task_5_func.get_returns_by_weight(prices, rebalance_date_list, weight_qp_dict[gamma])
    result = task_5_func.get_result(money_list, returns_list, weight_qp_dict[gamma], natural_days, H)
    data.append(result)

# SOCP 策略（各 sigma_max）
for sigma_max in sigma_max_list:
    money_list, returns_list = task_5_func.get_returns_by_weight(prices, rebalance_date_list, weight_socp_dict[sigma_max])
    result = task_5_func.get_result(money_list, returns_list, weight_socp_dict[sigma_max], natural_days, H)
    data.append(result)

# 策略名称列表
strategy_names = (["Equal Weight"] +
                  [f"QP(gamma={gamma})" for gamma in gamma_list] +
                  [f"SOCP(sigma_max={sm})" for sm in sigma_max_list])

# 输出绩效汇总表
df = pd.DataFrame(data, index=strategy_names)
print("\n")
print(df.to_string())

# 绘制部分 QP 策略的资产权重堆叠图（随时间变化）
gamma_list_plot = [0.1, 1, 10, 100]
asset_names = pd.read_csv('cleaned_assets.csv', header=None)[0].tolist()

for gamma in gamma_list_plot:
    weights = np.array(weight_qp_dict[gamma])
    max_weights = np.max(weights, axis=0)
    keep_assets = (max_weights >= 0.01)   # 保留最大权重大于1%的资产

    kept_asset_names = [asset_names[i] for i, ifkeep in enumerate(keep_assets) if ifkeep]
    labels = kept_asset_names + ['others']

    weights_main = weights[:, keep_assets]
    weights_others = weights[:, ~keep_assets].sum(axis=1)
    weights_plot = np.column_stack([weights_main, weights_others])

    plt.figure()
    plt.stackplot(range(len(rebalance_date_list) - 1), weights_plot.T, labels=labels, alpha=0.8)
    plt.xlabel('Date')
    plt.ylabel('Asset Weight')
    plt.title(f'Portfolio Weights(QP, gamma={gamma})')
    plt.legend(ncol=3, fontsize=7)
    plt.grid(linestyle='--', alpha=0.5)

plt.show()