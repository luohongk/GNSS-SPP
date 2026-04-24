<!--
 * @Author: Hongkun Luo
 * @Date: 2024-04-08 01:27:27
 * @LastEditors: luohongk luohongkun@whu.edu.cn
 * @Description: 
 * 
 * Hongkun Luo
-->

<h1 align="center">GNSS-SPP: A Python Object-Oriented GNSS Pseudorange Single Point Positioning System</h1>
  <h3 align="center">
    <a href="https://luohongkun.top/">Hongkun Luo</a>
  </h3>
  <p align="center">
    <a href="https://github.com/luohongk/GNSS-SPP">Project Repository</a>
  </p>
  <p align="center">
      <a href="https://github.com/luohongk/PseudorangeSPP">
          <img src="https://img.shields.io/badge/GNSS-SPP-blue" />
      </a>
      <a href="https://www.python.org/">
          <img src="https://img.shields.io/badge/Python-3.8+-blue" />
      </a>
      <a href="https://www.pyqtgraph.org/">
          <img src="https://img.shields.io/badge/GUI-PyQt5-green" />
      </a>
      <a href="https://www.igs.org/wg/rinex/">
          <img src="https://img.shields.io/badge/Format-RINEX_2.11-orange" />
      </a>
      <a href="https://www.gnu.org/licenses/gpl-3.0.html">
          <img src="https://img.shields.io/badge/License-GPL3.0-yellow.svg" />
      </a>
      <a href="https://www.zhiyuteam.com/">
          <img src="https://img.shields.io/badge/Wuhan University-BRAIN_LAB-green" />
      </a>
  </p>
</p>

<div align=center><img src="img/logo.png" width=100%></div>

# News

- **April 2025**: Added PyQt5 GUI with real-time visualization, outlier rejection, and statistics panel.
- **April 2024**: Released base code (command-line version).

# Demo

### GUI 界面

启动图形界面，选择观测文件与导航文件，点击「开始解算」即可实时查看定位结果：

```bash
python gui.py
```

图形界面提供四个标签页：
- **结果表格**：逐历元展示 ECEF 坐标（X/Y/Z）、迭代次数、有效卫星数
- **XYZ 时序**：三分量时序折线图，自动标注粗差历元（红叉）
- **平面散点**：水平偏差散点图，带粗差阈值圆
- **误差时序**：各历元 3D 定位偏差与 3D 中误差

### 命令行模式

```bash
python main.py
```

结果打印到终端，格式示例：

```
平差后的X坐标: -2148744.399  平差后的Y坐标: 4426641.032  平差后的Z坐标: 4044655.687  (迭代次数: 4, 有效卫星: 8颗)
```

# 1 Introduction

本项目基于 GPS 广播星历（RINEX 2.11 格式）实现 **GNSS 伪距单点定位（SPP）** 的 Python 面向对象版本。程序读取观测文件（`.xxo`）与导航文件（`.xxn`），逐历元进行最小二乘平差，解算接收机 ECEF 坐标。

**核心功能：**

| 模块 | 说明 |
|------|------|
| 广播星历解析 | 解析 RINEX 2.11 导航文件，完整实现开普勒轨道根数积分 |
| 卫星位置计算 | 支持偏近点角迭代、摄动改正、相对论钟差改正 |
| 电离层改正 | Klobuchar 广播电离层模型（ION ALPHA/BETA 来自导航文件头） |
| 对流层改正 | 完整 Saastamoinen 模型（干湿分量+高度修正） |
| 地球自转改正 | Sagnac 效应改正（信号传播期间地球自转） |
| 加权最小二乘 | 仰角相关权函数，支持迭代收敛（最大 50 次） |
| 粗差剔除 | 后验残差检验（阈值 30 m）+ GUI 层 10% 百分位剔除 |
| 仰角截止角 | 命令行 15°，GUI 模式 10° |
| PyQt5 可视化 | 多标签图形界面，后台线程解算，实时进度条 |

# 2 Dependencies

### 2.1 Python 版本

Python 3.8 及以上。

### 2.2 安装依赖

```bash
pip install numpy matplotlib PyQt5
```

| 包名 | 用途 |
|------|------|
| `numpy` | 矩阵运算、最小二乘求解 |
| `matplotlib` | 结果可视化（嵌入 PyQt5） |
| `PyQt5` | 图形界面框架 |

> **注意**：命令行模式（`main.py`）仅需 `numpy`，无需 `matplotlib` 和 `PyQt5`。

# 3 How does it work?

### 3.1 克隆项目

```bash
git clone https://github.com/luohongk/PseudorangeSPP.git
cd PseudorangeSPP
```

### 3.2 准备数据

项目 `data/` 目录中已附带示例数据（AL2H 测站，2023 年第 334 天）：

```
data/
├── al2h3340.23o    # RINEX 2.11 观测文件
└── al2h3340.23n    # RINEX 2.11 导航文件
```

如需使用自己的数据，将对应的 `.xxo` 和 `.xxn` 文件放入 `data/` 目录，并修改 `main.py` 中的文件路径，或在 GUI 中通过「浏览」按钮直接选择。

### 3.3 运行命令行版本

```bash
python main.py
```

### 3.4 运行图形界面版本

```bash
python gui.py
```

启动后点击「载入示例数据」可一键加载 `data/` 目录下的示例文件，再点击「▶ 开始解算」即可。

# 4 File Structure

| 文件名 | 功能 |
|--------|------|
| `main.py` | 命令行入口，调用各模块完成完整解算流程 |
| `readfile.py` | `ReadFile` 类：解析 RINEX O/N 文件，提取观测值、星历参数、电离层系数 |
| `satelite.py` | `Satelite` 类：基于广播星历计算卫星 ECEF 坐标和钟差 |
| `position.py` | `Position` 类：逐历元匹配卫星、施加延迟改正、迭代加权最小二乘定位 |
| `gui.py` | PyQt5 图形界面，多线程解算 + matplotlib 实时绘图 |
| `data/` | 示例 RINEX 数据（AL2H 测站） |

# 5 Positioning Principle

### 5.1 观测方程

对每一个卫星 $s$ 的伪距观测值 $R$：

$$
R = \sqrt{(x_s - x_r)^2 + (y_s - y_r)^2 + (z_s - z_r)^2} + c \cdot \delta t_r - c \cdot \delta t^s + d_{\text{ion}} + d_{\text{trop}}
$$

其中 $x_r, y_r, z_r$ 为接收机坐标，$\delta t_r$ 为接收机钟差，$\delta t^s$ 为卫星钟差，$d_{\text{ion}}$、$d_{\text{trop}}$ 分别为电离层和对流层延迟。

### 5.2 线性化（泰勒展开）

$$
V = B \cdot x - L
$$

设计矩阵 $B$ 的每一行为方向余弦加钟差系数：

$$
B_i = \left[ \frac{-(x_s - x_r^0)}{\rho_0},\ \frac{-(y_s - y_r^0)}{\rho_0},\ \frac{-(z_s - z_r^0)}{\rho_0},\ 1 \right]
$$

常数项：

$$
L_i = R_i - \rho_0 + c \cdot \delta t^s - d_{\text{ion}} - d_{\text{trop}}
$$

### 5.3 加权最小二乘解

$$
x = (B^T W B)^{-1} (B^T W L)
$$

权阵 $W$ 根据卫星仰角 $\theta$ 构造：

$$
\sigma_i = 0.003 + \frac{0.003}{\sin\theta_i}, \quad w_i = \frac{1}{\sigma_i^2}
$$

### 5.4 迭代更新

$$
\begin{cases} x_r = x_r^0 + \Delta x_r \\ y_r = y_r^0 + \Delta y_r \\ z_r = z_r^0 + \Delta z_r \end{cases}
$$

当坐标改正量 $\sqrt{\Delta x_r^2 + \Delta y_r^2 + \Delta z_r^2} < 0.001\ \text{m}$ 时收敛。
