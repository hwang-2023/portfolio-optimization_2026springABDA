# task_4_func.py
import numpy as np

# 使用 PDHG (Primal-Dual Hybrid Gradient) 求解 A2 模型
# 输入: gamma (风险厌恶系数), tau, xi (步长参数), sigma, mu
# 输出: 最优解 x, 目标函数值序列 f_list, 原始残差序列 r_primal_list,
#       对偶残差序列 r_dual_list, 实际迭代次数 it
def solve_a2_by_pdhg(gamma, tau, xi, sigma, mu):
    n = len(mu)

    # 预先计算矩阵求逆，用于 x 更新
    a = np.linalg.inv(gamma * tau * sigma + np.eye(n))

    # 初始化原始变量和对偶变量
    x = np.ones(n) / n
    x_hat = x.copy()
    p = 0          # 对偶变量对应于等式约束 sum(x)=1
    q = np.zeros(n) # 对偶变量对应于不等式约束 x>=0

    it = 99999
    f_list = []
    r_primal_list = []
    r_dual_list = []

    for k in range(100000):
        x_prev = x.copy()

        # 对偶变量更新（上升步）
        p = p + xi * (sum(x_hat) - 1)
        q = np.maximum(q - xi * x_hat, np.zeros(n))

        # 原始变量更新（下降步）
        x = a @ (x - tau * (p * np.ones(n) - q) + tau * mu)

        # 外推步
        x_hat = x + x - x_prev

        # 计算原始残差（等式约束违反程度 + 不等式约束违反程度）
        r_primal = abs(sum(x) - 1) + np.linalg.norm(np.minimum(x, 0))
        # 计算对偶残差（最优性条件的残差）
        r_dual = np.linalg.norm(gamma * sigma @ x - mu + p * np.ones(n) - q)

        r_primal_list.append(r_primal)
        r_dual_list.append(r_dual)
        f_list.append(gamma * x.T @ sigma @ x * 0.5 - mu.T @ x)

        # 收敛判断
        if r_primal < 1e-4 and r_dual < 1e-4:
            it = k
            break

    return x, f_list, r_primal_list, r_dual_list, it