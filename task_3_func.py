# task_3_func.py
import numpy as np

# 投影到单纯形（非负且和为1）
# 输入: 向量 y
# 输出: 投影向量 p
def proj_simplex(y):
    n = len(y)
    v = np.sort(y)[::-1]   # 降序排序
    p = np.zeros(n)
    s = 0
    # 寻找阈值 lambda 使得投影后非负且和为1
    for k in range(1, n + 1, 1):
        if k == n:
            lam = (s - 1) / k + v[n - 1]
            p = y - lam * np.ones(n)
            break
        s += k * (v[k - 1] - v[k])
        if s >= 1:
            lam = (s - 1) / k + v[k]
            p = np.maximum(y - lam * np.ones(n), 0)
            break
    return p

# 使用 ADMM 求解 A2 模型
# 输入: gamma (风险厌恶系数), rho (ADMM 惩罚参数), sigma, mu
# 输出: 最优解 x, 目标函数值序列 f_list, 原始残差序列 r_primal_list,
#       对偶残差序列 r_dual_list, 实际迭代次数 it
def solve_a2_by_admm(gamma, rho, sigma, mu):
    n = len(mu)

    # 预先计算矩阵求逆，用于 x 更新步骤
    a = np.linalg.inv(gamma * sigma + rho * np.eye(n))

    # 初始化变量
    x = np.ones(n) / n
    z = np.ones(n) / n
    u = np.zeros(n)

    it = 9999
    f_list = []
    r_primal_list = []
    r_dual_list = []

    for k in range(10000):
        z_prew = z.copy()
        # x 更新
        x = a @ (mu + rho * (z - u))
        # z 更新：投影到单纯形
        z = proj_simplex(x + u)
        # 对偶变量更新
        u += x - z

        # 计算残差
        r_primal = np.linalg.norm(x - z)
        r_dual = rho * np.linalg.norm(z - z_prew)
        r_primal_list.append(r_primal)
        r_dual_list.append(r_dual)
        f_list.append(gamma * x.T @ sigma @ x * 0.5 - mu.T @ x)

        # 收敛判断
        if r_primal < 1e-4 and r_dual < 1e-4:
            it = k
            break

    return x, f_list, r_primal_list, r_dual_list, it