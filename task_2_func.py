# task_2_func.py
import cvxpy as cp
import numpy as np
from scipy.linalg import cholesky

# 使用 CVXPY 求解 A2 模型（均值-方差优化，风险厌恶系数 gamma）
# 输入: gamma (风险厌恶系数), sigma (协方差矩阵), mu (期望收益向量)
# 输出: 最优权重向量 x
def solve_a2_by_cvx(gamma, sigma, mu):
    n = len(mu)
    x = cp.Variable(n)

    # 目标函数：最小化 gamma/2 * x^T Sigma x - mu^T x
    objective = cp.Minimize(gamma * 0.5 * cp.quad_form(x, sigma) - mu @ x)

    # 约束：非负权重，和为1
    constraints = [x >= 0,
                   cp.sum(x) == 1]

    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.MOSEK)

    print(f"\ncvx 求解 A2 model, gamma={gamma}")
    print("求解状态:", prob.status)
    print("最优目标函数值:", result)
    if prob.status == 'optimal':
        print("最优解前 4 个值:\n", x.value[:4])

    return x.value

# 使用 CVXPY 求解 B 模型（在给定风险上限 sigma_max 下最大化收益）
# 输入: sigma (协方差矩阵), mu (期望收益向量), sigma_max (风险上限)
# 输出: 最优权重向量 x
def solve_b_by_cvx(sigma, mu, sigma_max):
    n = len(mu)
    # 对协方差矩阵进行 Cholesky 分解，用于构造二阶锥约束
    b = cholesky(sigma, lower=True)

    x = cp.Variable(n)
    # 目标：最大化 mu^T x
    objective = cp.Maximize(mu @ x)
    # 约束：||B^T x||_2 <= sigma_max, x >= 0, sum(x) = 1
    constraints = [cp.norm(b.T @ x, 2) <= sigma_max,
                   x >= 0,
                   cp.sum(x) == 1]

    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.MOSEK)

    print(f"\ncvx 求解 B model, sigma_max={sigma_max}")
    print("求解状态：", prob.status)
    print("最优目标函数值：", result)
    if prob.status == 'optimal':
        print("最优解前 4 个值:\n", x.value[:4])

    return x.value