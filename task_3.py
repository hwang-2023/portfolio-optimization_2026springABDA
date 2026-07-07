# task_3.py
import task_3_func
import task_2_func
import numpy as np
import matplotlib.pyplot as plt
import time

# 加载 mu 和 sigma
loaded = np.load('mu_n_sigma.npz')
mu = loaded['mu']
sigma = loaded['sigma']

gamma = 1
# 不同的 rho 参数（ADMM 惩罚参数）
rho_list = [0.005, 0.01, 0.02, 0.05, 0.1]

it_list = []      # 迭代次数
time_list = []    # 运行时间
f_opt_list = []   # 最终目标函数值

# 对每个 rho 运行 ADMM 求解 A2 模型
for rho in rho_list:
    start = time.perf_counter()
    x, f_list, r_primal_list, r_dual_list, it = task_3_func.solve_a2_by_admm(gamma, rho, sigma, mu)
    end = time.perf_counter()
    time_list.append(end - start)
    it_list.append(it)
    f_opt_list.append(f_list[-1])

    # 绘制目标函数值变化曲线
    plt.figure(1)
    plt.plot(range(it + 1), f_list, label=f"gamma={gamma},rho={rho}")
    # 绘制原始残差变化
    plt.figure(2)
    plt.plot(range(it + 1), r_primal_list, label=f"gamma={gamma},rho={rho}")
    # 绘制对偶残差变化
    plt.figure(3)
    plt.plot(range(it + 1), r_dual_list, label=f"gamma={gamma},rho={rho}")
    # 绘制最终权重分布（散点图）
    plt.figure(4)
    plt.scatter(range(len(x)), x, label=f"gamma={gamma},rho={rho}", s=5)

# 使用 CVXPY 求解作为对比基准
start = time.perf_counter()
x = task_2_func.solve_a2_by_cvx(gamma, sigma, mu)
end = time.perf_counter()
time_list.append(end - start)
f = gamma * x.T @ sigma @ x * 0.5 - mu.T @ x
f_opt_list.append(f)

# 输出结果汇总
print("\nrho=", rho_list, "cvx")
print("运行时间: ", time_list)
print("\nrho=", rho_list, "cvx")
print("最优值: ", f_opt_list)
print("\nrho=", rho_list)
print("迭代次数: ", it_list)

# 在目标函数图中添加 CVX 的参考线
plt.figure(1)
plt.plot([0, max(it_list)], f * np.ones(2), label=f"gamma={gamma},cvx")
# 在权重图中添加 CVX 的结果
plt.figure(4)
plt.scatter(range(len(x)), x, label=f"gamma={gamma},cvx", s=5)

# 设置各图标题和图例
plt.figure(1)
plt.title('f_list')
plt.legend()

plt.figure(2)
plt.title('r_primal_list')
plt.legend()

plt.figure(3)
plt.title('r_dual_list')
plt.legend()

plt.figure(4)
plt.title('x_opt')
plt.legend()

plt.show()