# Portfolio Optimization with ADMM & PDHG

基于Markowitz均值-方差框架的投资组合优化项目，实现了QP和SOCP两种优化模型，使用ADMM和PDHG两种算法求解，并完成了滚动回测与绩效评估。

## 技术栈
- Python, NumPy, Pandas
- CVXPY (MOSEK solver)
- Matplotlib
- Tiingo API (数据获取)

## 项目结构
├── task_1/          # 数据获取与清洗（S&P 500成分股，2016-2025）
├── task_2/          # 有效前沿绘制（QP + SOCP模型）
├── task_3/          # ADMM算法实现
├── task_4/          # PDHG算法实现
└── task_5/          # 滚动回测（L=252, H=20）与绩效评估

## 核心功能
1. **数据工程**：从Wikipedia获取S&P 500成分股，通过Tiingo API下载100只股票10年日频数据
2. **优化模型**：实现均值-方差优化的QP模型（风险厌恶系数γ）和SOCP模型（风险上限σ_max）
3. **算法实现**：手写ADMM和PDHG两种优化算法，与CVXPY基准对比
4. **回测系统**：252天滚动窗口估计，每20个交易日调仓，包含夏普比率、最大回撤、换手率等指标

## 运行方式
```bash
python task_1.py   # 下载并清洗数据
python task_2.py   # 绘制有效前沿
python task_3.py   # ADMM求解与对比
python task_4.py   # PDHG求解与对比
python task_5.py   # 滚动回测
