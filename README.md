# pypile（桥梁基础结构空间静力分析程序）

<div align="center">

![版本](https://img.shields.io/badge/版本-1.0.2-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![许可证](https://img.shields.io/badge/许可证-GPL--3.0-orange)

</div>

## 📋 项目概述

PyPile 是一个用于桥梁基础结构空间静力分析的 Python 包，其源代码(BCAD_PILE)由 Fortran 代码转换而来。该工具可以执行桩基础在不同荷载条件下的行为分析，包括位移、内力以及土-结构相互作用。能够生成报告, 并自动推算最不利单桩内力。

## ✨ 主要功能

- 桩基础的空间静力分析
- 桩基变形因子计算
- 轴向和横向刚度分析
- 桩基内力和位移计算
- 分析结果输出报告 ✅
- 基于 Plotly 的交互式 3D 可视化 (TODO)

## 📦 安装

### 通过 uv 安装

```bash
# 若未安装 uv，可先安装 uv
pip install uv
# 安装 pypile 工具(希望在命令行中独立使用请这样安装)
uv tool install pypile
# 安装 pypile 包(希望在Python脚本中使用请这样安装)
uv install pypile
```

### 使用 pip 安装

```bash
pip install pypile
```

## 🔧 依赖项

- art>=6.4
- loguru>=0.7.3
- matplotlib>=3.7.5
- numpy>=1.24.4
- pydantic>=2.10.6
- tabulate>=0.9.0

## 📘 使用方法

### 命令行界面

```bash
usage: pypile [-h] [-f FILE] [-s] [-db] [-p] [-d] [-o] [-v] [-force FORCE FORCE FORCE FORCE FORCE FORCE] [-mode {replace,add}]

PyPile - 桩基础分析程序

选项:
  -h, --help            显示帮助信息并退出
  -f FILE, --file FILE  指定输入数据文件（.dat格式）
  -s, --select          通过文件选择器选择计算(✅)/验算(TODO)文件
  -db, --debug          启用调试模式，输出详细日志
  -p, --print           打印计算结果摘要
  -d, --detail          打印详细计算结果
  -o, --old             运行旧版BCAD_PILE程序
  -v, --version         显示程序版本号并退出
  -force FORCE FORCE FORCE FORCE FORCE FORCE, --force FORCE FORCE FORCE FORCE FORCE FORCE
                        指定作用在(0,0)点的力 [FX, FY, FZ, MX, MY, MZ]
  -mode {replace,add}, --mode {replace,add}
                        设置力的作用模式：replace（替换）或add（添加）
```

### 🌟 终端命令行示例（使用`uv tool install`安装）

终端命令帮助可直接使用`pypile -h`查看

**Case 1**:

```bash
# 直接运行，在后续的提示中输入文件路径进行默认分析
pypile
```

**Case 2**:

```bash
# 指定输入文件路径进行分析
pypile -f ./tests/Test-1-2.dat
```

**Case 3**:

```bash
# 指定输入文件路径并打印结果摘要
pypile -f ./tests/Test-1-2.dat -p
```

**Case 4**:

```bash
# 指定输入文件路径并打印详细结果
pypile -f ./tests/Test-1-2.dat -p -d
```

**Case 5**:

```bash
# 使用文件选择框选择文件，并打印结果
pypile -s
```

**Case 6**:

```bash
# 使用原BCAD_PILE程序进行分析
pypile -o
```

**Case 7**:

```bash
# 进行debug
pypile -db
```

**Case 8**:

```bash
# 显示版本信息
pypile -v
```


### 🌟 Python API 示例

```python
from pypile import PileManager
from pathlib import Path
import numpy as np

# 初始化桩基管理器
pile = PileManager()
# 读取数据文件
pile.read_dat(Path("./tests/Test-1-2.dat"))
    
# 设置NumPy输出格式
np.set_printoptions(linewidth=200, precision=2, suppress=True)
# 查看基础刚度
print(f"Pile stiffness matrix K:\n{pile.K}")
# 查看按照Sap2000中Coupled Spring的刚度格式
print(f"Pile stiffness matrix K_SAP:\n{pile.K_SAP}")

# 查看指定桩的刚度
# ino: int = 5
# print(f"Pile {ino} stiffness matrix:\n{pile.K_pile(ino)}")
    
# 设置荷载
force = np.array([22927.01, 0, 40702.94, 0.0, 320150.23, 0])
np.set_printoptions(linewidth=200, precision=4, suppress=True)
# 获取承台位移
print(f"Cap displacement:\n{pile.disp_cap(force)}")
# 获取各桩桩顶位移
# print(f"Pile displacement:\n{pile.disp_piles(force)}")

# 计算基础反力，得到一个Dict，key是桩号(int)，value是PileResult对象
pile_results = pile.eforce(force)

# 生成刚度矩阵报告
pile.stiffness_report()
# 生成群桩基础报告
pile.pile_group_report()
# 生成最不利单桩报告
pile.worst_pile_report()

# 获取最不利单桩结果
print(pile.worst_pile_force)
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
-1.5 -1.5 %桩的坐标(x，y)
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

### 终端输出示例

```
(∩｀-´)⊃━☆ﾟ.*･｡ﾟ  **正在读取输入信息**  【シ】

(∩｀-´)⊃━☆ﾟ.*･｡ﾟ  **计算桩的变形因子**  [^_^]

(∩｀-´)⊃━☆ﾟ.*･｡ﾟ  **计算桩的轴向刚度**  ^‿^

(∩｀-´)⊃━☆ﾟ.*･｡ﾟ  **计算桩的侧向刚度**  ( ͡ʘ ͜ʖ ͡ʘ)

㋡      **计算桩基承台的位移和内力**    [^_^]

程序运行完成，刚度矩阵、群桩及最不利单桩报告已保存到 D:\pypile\tests\Test-1-1.out，所有桩验算结果已保存到 D:\pypile\tests\Test-1-1.pos。ಠ◡ಠ
```

## 📄 输出报告格式

pypile 生成两种输出报告文件：

1. `.out` 文件 - 包含刚度矩阵、群桩报告和最不利单桩报告
2. `.pos` 文件 - 包含所有桩的验算结果详情

输出报告示例：

```
-----------------------------------------------------------------------
|    _ (`-.                    _ (`-.                          ('-.    |
|   ( (OO  )                  ( (OO  )                       _(  OO)   |
|  _.`     \   ,--.   ,--.   _.`     \   ,-.-')   ,--.      (,------.  |
| (__...--''    \  `.'  /   (__...--''   |  |OO)  |  |.-')   |  .---'  |
|  |  /  | |  .-')     /     |  /  | |   |  |  \  |  | OO )  |  |      |
|  |  |_.' | (OO  \   /      |  |_.' |   |  |(_/  |  |`-' | (|  '--.   |
|  |  .___.'  |   /  /\_     |  .___.'  ,|  |_.' (|  '---.'  |  .--'   |
|  |  |       `-./  /.__)    |  |      (_|  |     |      |   |  `---.  |
|  `--'         `--'         `--'        `--'     `------'   `------'  |
|                                                                      |
|                                                  Version:1.0.0, 2025 |
|                                                      By: Lingyun Gou |
-----------------------------------------------------------------------
```

后续内容包括桩基础刚度矩阵、群桩位移和内力计算结果以及最不利单桩验算结果。

## 🏗️ 项目结构

```
pypile/
├── pypile/
│   ├── __init__.py
│   ├── cli.py             # 命令行接口
│   ├── pile_manager.py    # 桩基管理核心类
│   ├── report.py          # 报告生成功能
│   ├── models/
│   │   ├── __init__.py
│   │   ├── arrange_model.py     # 排布模型
│   │   ├── control_model.py     # 控制模型
│   │   ├── no_simu_model.py     # 非模拟桩模型
│   │   ├── pile_parser.py       # 桩基解析器
│   │   ├── pile_results_model.py # 计算结果模型
│   │   └── simu_pile_model.py   # 模拟桩模型
│   └── original/          # 原始资源和参考
├── tests/                 # 测试文件
├── docs/                  # 文档
├── examples/              # 示例文件
├── dist/                  # 分发文件
├── pyproject.toml         # 项目配置
├── LICENSE
└── README.md
```

## 👥 贡献指南

欢迎对 PyPile 项目做出贡献！请积极提交 Issue，或者提交 Pull Request。

## 📜 许可证

GPL-3.0 许可证 - 详情请参阅 `LICENSE` 文件

## 🙏 致谢

转换自同济大学桥梁工程系 CAD 研究组开发的原始 Fortran BCAD_PILE 程序。
