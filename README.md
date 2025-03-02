# pypile（桥梁基础结构空间静力分析程序）

<div align="center">

![版本](https://img.shields.io/badge/版本-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.7+-green)
![许可证](https://img.shields.io/badge/许可证-GPL--3.0-orange)

</div>

## 📋 项目概述

pypile 是一个用于桥梁基础结构空间静力分析的 Python 包，其源代码(BCAD_PILE)由 Fortran 代码转换而来。该工具可以执行桩基础在不同荷载条件下的行为分析，包括位移、内力以及土-结构相互作用。特别适用于桥梁结构在地震及多灾害(如冲刷、液化等)条件下的基础分析。

## ✨ 主要功能

- 桩基础的空间静力分析
- 桩基变形因子计算
- 轴向和横向刚度分析
- 桩基内力和位移计算
- 分析结果可视化
- 基于 Plotly 的交互式 3D 可视化

## 📦 安装

### 使用 pip 安装

```bash
pip install pypile
```

### 从源代码安装

```bash
git clone https://github.com/ganansuan647/pypile.git
cd pypile
pip install -e .
```

## 🔧 依赖项

- Python 3.7+
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- Numba >= 0.53.0 (性能优化)
- Plotly >= 5.0.0 (交互式可视化)

## 📘 使用方法

### 命令行界面

```bash
# 基本分析
bcad_pile input_file.dat

# 带可视化的分析
bcad_pile input_file.dat --visualize
```

### Python API

```python
from bcad_pile.core.computation import analyze_pile_foundation, extract_visualization_data
from bcad_pile.visualization.plotter import plot_results

# 运行分析
results = analyze_pile_foundation("input_file.dat")

# 创建可视化
vis_data = extract_visualization_data(results)
plot_results(vis_data)

# 创建交互式可视化
from bcad_pile.visualization.interactive_view import create_interactive_visualization
fig = create_interactive_visualization(vis_data)
fig.show()
```

## 📄 输入文件格式

BCAD_PILE 使用与原始 Fortran 实现相同的输入文件格式。基本结构包括:

```
[contral]
2 %1为计算位移、内力  2为计算桩基子结构的刚度 3为计算某一根桩的刚度
% 1 %外荷载的作用点数%
% 0 0 %作用点（x，y）%
% 0 9270 58697 250551.6 0 0 %外力，分别为x,y,z方向的力与弯矩，注意弯矩与剪力的对应正负，有右手法则判断%
end
[arrange]
4 0   %非虚桩 虚拟桩的根数%
-1.5 -1.5 %桩的坐标（x，y)
-1.5 1.5 
1.5 1.5 
1.5 -1.5 
end
[no_simu] %非虚拟单桩信息%
0 0 0 0  %控制信息，一般不改,大于根数。。%
<0>
0 1 0 0 1 %一、单桩的形状信息：1为方0为圆；二。支撑信息：1钻孔灌注2打入摩擦3端承非嵌固4端承嵌固； 三四五为x,y,z交角的余弦值%
0 0 0 0  %在土层上的桩：层数、桩长、外径、输出点数%
4 14.84 1.2 4e3 14 10   %在土层下的桩：4为土层层数，之后分别为第i段的桩长、外径、地基比例系数m（主要参考塑性），摩擦角（看土类），输出点数（1m一个）%
   5.0 1.2 1.2e4 20.3 10
   5.8 1.2 2.5e4 18 10
   24.51 1.2 5e4 30 10
3e4 3e7 1 %1摩擦桩的桩底比例系数活柱桩的地基系数 2桩身混凝土弹性模量3抗弯刚度折减系数，一般取1%
end
[simu_pe]
end
```

详细的输入文件说明可参考 `docs/input_format.md` 文件。

## 🏗️ 项目结构

```
pypile/
├── bcad_pile/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data.py         # 数据结构
│   │   ├── reader.py       # 输入文件读取
│   │   ├── writer.py       # 输出文件写入
│   │   ├── computation.py  # 计算流程控制
│   │   ├── stiffness.py    # 刚度计算
│   │   ├── displacement.py # 位移计算
│   │   └── forces.py       # 内力计算
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── matrix.py       # 矩阵操作
│   │   └── math_helpers.py # 数学辅助函数
│   └── visualization/
│       ├── __init__.py
│       ├── plotter.py      # 静态可视化
│       └── interactive_view.py # 交互式可视化
├── tests/                  # 测试文件
├── examples/               # 示例文件
├── docs/                   # 文档
├── setup.py
├── LICENSE
└── README.md
```

## 🌟 示例

### 基本分析示例

```python
from bcad_pile.core.computation import analyze_pile_foundation

# 使用样例输入文件进行分析
results = analyze_pile_foundation("examples/example1.dat")

# 输出桩基础的整体刚度矩阵
print("Foundation Stiffness Matrix:")
print(results['stiffness_matrix'])

# 输出第一根桩的位移
print("Displacements of Pile 1:")
print(results['pile_results'][0]['top_displacement'])
```

### 可视化示例

```python
from bcad_pile.core.computation import analyze_pile_foundation, extract_visualization_data
from bcad_pile.visualization.plotter import plot_results

# 分析并可视化
results = analyze_pile_foundation("examples/example2.dat")
vis_data = extract_visualization_data(results)
plot_results(vis_data)
```

## 🌊 在多灾害分析中的应用

BCAD_PILE 特别适合桥梁基础在复合灾害条件下的分析，包括：

- **地震作用**：分析地震荷载下桩基础的响应
- **冲刷影响**：模拟河床冲刷对桩基础稳定性的影响
- **土壤液化**：评估土壤液化对桩基础承载力的削弱
- **荷载组合**：分析多种灾害同时作用下的桩基础行为

## 👥 贡献指南

欢迎对 BCAD_PILE 项目做出贡献！请参阅 `CONTRIBUTING.md` 文件了解贡献流程。

## 📜 许可证

GPL-3.0 许可证 - 详情请参阅 `LICENSE` 文件

## 🙏 致谢

转换自同济大学桥梁工程系 CAD 研究组开发的原始 Fortran BCAD_PILE 程序。
