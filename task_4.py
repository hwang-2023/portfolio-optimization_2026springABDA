# task_4.py
import task_4_func
import task_2_func
import numpy as np
import matplotlib.pyplot as plt
import time

# 加载 mu 和 sigma
loaded = np.load('mu_n_sigma.npz')
mu = loaded['mu']
sigma = loaded['sigma']
n = len(mu)

gamma = 1
# 定义 PDHG 算法参数
tau_list = [0.2, 0.5, 1, 2]      # 步长参数 tau
xi_list = [0.002, 0.005, 0.01, 0.02]   # 步长参数 xi

it_list = []
time_list = []
f_opt_list = []
s_list = []   # 存储参数组合标签

# 遍历所有参数组合，仅当满足收敛条件 0.2 < tau * xi * (n+1) < 1 时运行
for tau in tau_list:
    for xi in xi_list:
        if 0.2 < tau * xi * (n + 1) < 1:
            start = time.perf_counter()
            x, f_list, r_primal_list, r_dual_list, it = (
                task_4_func.solve_a2_by_pdhg(gamma, tau, xi, sigma, mu))
            end = time.perf_counter()
            time_list.append(end - start)
            it_list.append(it)
            f_opt_list.append(f_list[-1])
            s_list.append(f"gamma={gamma},tau={tau},xi={xi}")

            # 绘制目标函数值、原始残差、对偶残差和最终权重
            plt.figure(1)
            plt.plot(range(it + 1), f_list, label=f"gamma={gamma},tau={tau},xi={xi}")
            plt.figure(2)
            plt.plot(range(it + 1), r_primal_list, label=f"gamma={gamma},tau={tau},xi={xi}")
            plt.figure(3)
            plt.plot(range(it + 1), r_dual_list, label=f"gamma={gamma},tau={tau},xi={xi}")
            plt.figure(4)
            plt.scatter(range(len(x)), x, label=f"gamma={gamma},tau={tau},xi={xi}", s=5)

            print(f"gamma={gamma}, tau={tau}, xi={xi}已完成")

# CVXPY 基准求解
start = time.perf_counter()
x = task_2_func.solve_a2_by_cvx(gamma, sigma, mu)
end = time.perf_counter()
f = gamma * x.T @ sigma @ x / 2 - mu.T @ x

# 在图中添加 CVX 结果作为参考
plt.figure(1)
plt.plot([0, max(it_list)], f * np.ones(2), label=f"gamma={gamma},cvx")
plt.figure(4)
plt.scatter(range(len(x)), x, label=f"gamma={gamma},cvx", s=5)

# 打印所有 PDHG 参数组合的结果
for i in range(len(s_list)):
    print(f"\n{s_list[i]}:")
    print(f"运行时间: {time_list[i]}, 迭代次数: {it_list[i]}, 最优值: {f_opt_list[i]}")

print(f"\ngamma={gamma},cvx:")
print(f"迭代次数: {end - start}, 最优值: {f}")

# 设置各图标题和图例
plt.figure(1)
plt.title('f_list')
plt.legend(fontsize=8)

plt.figure(2)
plt.title('r_primal_list')
plt.legend(fontsize=8)

plt.figure(3)
plt.title('r_dual_list')
plt.legend(fontsize=8)

plt.figure(4)
plt.title('x_opt')
plt.legend(fontsize=8)

plt.show()