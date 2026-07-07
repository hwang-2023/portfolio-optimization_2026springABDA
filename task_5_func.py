# task_5_func.py
import pandas as pd
import numpy as np

# 根据给定的权重序列和调仓日期，计算资金曲线和每期收益率
# 输入: price (价格DataFrame), r_d_l (调仓日期列表), w (权重列表，每个元素为当期权重向量)
# 输出: money_list (每期期末资金), returns_list (每期收益率)
def get_returns_by_weight(price, r_d_l, w):
    money_list = [10000]      # 初始资金
    returns_list = []
    m = len(r_d_l)
    for i in range(m - 1):
        # 调仓日买入：按当期权重分配资金
        buy = money_list[-1] * w[i]

        # 下一个调仓日的前一天为卖出日
        sold_date_idx = price.index.get_loc(r_d_l[i + 1]) - 1
        sold_date = price.index[sold_date_idx]

        # 卖出后的资金总额
        sold = buy / price.loc[r_d_l[i]] * price.loc[sold_date]
        money_list.append(np.sum(sold))
        returns_list.append((np.sum(sold) - np.sum(buy)) / np.sum(buy))

    return money_list, returns_list

# 根据资金曲线和收益率序列计算各项绩效指标
# 输入: money_list (资金序列), returns_list (每期收益率), weight_list (权重序列),
#       natural_days (总自然天数), h (调仓间隔天数)
# 输出: 字典，包含累计财富、累计收益率、年化收益率、年化波动率、夏普比率、
#       最大回撤、平均换手率、不同交易成本下的最终资金
def get_result(money_list, returns_list, weight_list, natural_days, h):
    cumulative_wealth = money_list[-1] - money_list[0]
    cumulative_return = cumulative_wealth / money_list[0]
    annualized_return = (1 + cumulative_return) ** (365 / natural_days) - 1
    annualized_volatility = np.std(returns_list, ddof=1) * np.sqrt(252 / h)

    sharpe_ratio = np.mean(returns_list) / np.std(returns_list, ddof=1) * np.sqrt(252 / h)

    # 最大回撤计算
    nav = np.cumprod(returns_list + np.ones(len(returns_list)))
    running_max = np.maximum.accumulate(nav)
    drawdown = (nav - running_max) / running_max
    max_drawdown = np.min(drawdown)

    # 换手率计算：每个调仓期的换手率为 0.5 * sum(|w_curr - w_prev|)
    turnover = [0.0]
    for i in range(len(weight_list) - 1):
        w_prev = weight_list[i]
        w_curr = weight_list[i + 1]
        turn = 0.5 * np.sum(np.abs(w_curr - w_prev))
        turnover.append(turn)
    avg_turnover = np.mean(turnover[1:])

    # 不同交易成本下的敏感性分析
    c_list = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    sensitivity_to_transaction_costs = []
    for c in c_list:
        money = 10000
        for r, turn in zip(returns_list, turnover):
            cost = c * turn
            money *= (1 + r) * (1 - cost)
        sensitivity_to_transaction_costs.append(round(money, 6))

    return {
        'cumulative_wealth': cumulative_wealth,
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_turnover': avg_turnover,
        'sensitivity_to_transaction_costs(0,0.1%,0.2%,0.5%,1%,2%,5%)': sensitivity_to_transaction_costs
    }