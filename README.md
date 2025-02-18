# pypile
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.8%2B-orange)](https://scipy.org)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

PyPile是一款用Python实现的专业化桥梁桩基分析工具,基于m法基本假设将任意布置和土层的桩基础简化为六弹簧，灵感来源于同济大学编写的BCAD_PILE程序。提供高性能的计算核心、现代化的可视化功能和友好的输入输出接口。

## 特性

- **高性能计算**:采用 NumPy + SciPy 进行矩阵运算,Numba JIT加速
- **丰富可视化**:基于 PyVista 实现交互式三维可视化
- **现代架构**:遵循模块化设计,提高可维护性
- **友好接口**:支持JSON/YAML配置和原生文本数据输入格式

## 安装

PyPile需要Python 3.8或更高版本。可通过pip安装:

```bash
pip install pypile
```

## 快速开始

运行内置的示例分析:

```bash
pypile examples/bridge_case.yml
```

该命令将执行标准的桥梁分析流程,对结果进行交互式三维渲染,并输出内力报告。

## 文档

完整的用户手册和API文档托管在项目网站: https://pypile.readthedocs.io/

## 贡献

欢迎为PyPile做出贡献!如有任何问题或建议,请先查看文档并遵循贡献指南。

## 许可

PyPile采用GPL-3.0 license许可证。

---

# PyPile 技术规范

## 1. 总体设计

PyPile遵循模块化设计原则,由以下核心组件构成:

- **ConfigParser**: 配置文件解析器,支持原生文本、JSON和YAML格式
- **PileBuilder**: 根据配置信息构建桩基模型对象
- **NonlinearSolver**: 实现迭代求解桩基变形、应力和内力等响应
- **ResultRenderer**: 将分析结果可视化为二维平面图和交互式三维场景

这些模块均实现为Python类,并由统一的`PileAnalyzer`控制器进行集成和协调。

## 2. 坐标系统

PyPile采用右手笛卡尔坐标系统,基准平面为桩顶平面,采用约定:

- Z轴垂直向下为正方向
- X轴平行承台边线
- Y轴根据右手法则确定

桩轴线方向由其在XYZ坐标系下的方向余弦定义。

## 3. 输入输出规范

### 3.1 配置文件格式

PyPile支持以下配置文件格式:

- **YAML**:标准的YAML版本,推荐使用
- **JSON**: JSON格式兼容
- **PileText**: 向后兼容历史文本数据格式

其中YAML/JSON配置文件格式高度结构化,示例如下:

```yaml
pile_groups:
  - type: non-virtual 
    coordinates: [10.2, 5.6]
    properties:
      diameter: 1.2
      segments:
        - {depth: 5, soil_type: clay}
        - {depth: 10, soil_type: sand}
  - type: virtual
    stiffness: [1e6, 1e6, 2e6, 0, 0, 0]
    
load_cases:
  - node: [0, 0, 0]
    forces: [0, 1000, 0, 0, 0, 0]
```

### 3.2 输出报告

分析结果将以以下格式输出:

- **PlainText**: 向后兼容的文本报告格式
- **MarkdownReport**: 支持排版的Markdown格式报告
- **Excel**: 具有可视化效果的Excel电子表格输出

## 4. 核心算法

PyPile的核心计算模块采用下述算法和方法:

### 4.1 单桩刚度计算

采用分段幂级数解析法和有限差分数值求解相结合的混合算法:

1. 自由段或均匀段采用解析解
2. 变截面段采用高阶有限差分求解

这确保了计算的高精度和高效性。

### 4.2 整体解析

整体有限元分析中,采用以下方法实现非线性迭代求解:

1. 系统刚度矩阵组装: 采用对角矩阵存储格式,提高计算效率
2. 控制方程求解: 改进的Newton-Raphson迭代法
3. 边界条件处理: 基于拉格朗日乘子法施加位移边界条件

可选择直接迭代或增量迭代两种策略。

### 4.3 并行加速

针对计算密集型模块,PyPile使用Numba实现JIT(Just-In-Time)编译,发挥CPU矢量化能力,实现高效并行计算。

## 5. 三维可视化

PyPile基于PyVista提供了交互式三维可视化功能,包括:

- **变形云图**: 整体模型形变状态的可视化
- **剖面视图**: 任意剖面方向的单桩内力分布
- **矢量图**: 单桩土压力分布矢量场

此外,用户还可以通过界面控制调整视角、缩放级别和渲染选项。

## 6. 拓展接口

PyPile提供了一系列钩子函数,方便用户扩展和定制功能:

- `register_pre_analysis_hook(func)`: 分析前处理钩子
- `register_post_analysis_hook(func)`: 分析后处理钩子
- `register_soil_model(name, func)`: 注册新的土层本构模型

详情请参考API文档中的`pypile.extensions`模块。

## 7. 版本控制

PyPile采用[语义化版本](https://semver.org/lang/zh-CN/)进行版本管理,版本号格式为:

```
主版本号.次版本号.修订号
```

版本更新日志记录在`CHANGELOG.md`文件中。

## ConfigParser 模块

`ConfigParser` 模块用于解析配置文件，支持 YAML、JSON 和文本格式。该模块包含以下类和方法：

- `ConfigParser` 类：协调配置文件解析任务
  - `parse(file_path)` 方法：根据文件格式调用相应的解析方法
  - `get_data()` 方法：返回解析后的数据
  - `validate_data()` 方法：验证解析后的数据

- `YamlParser` 类：处理 YAML 文件解析
  - `parse(file_path)` 方法：解析 YAML 文件并返回结构化数据

- `JsonParser` 类：处理 JSON 文件解析
  - `parse(file_path)` 方法：解析 JSON 文件并返回结构化数据

- `TextParser` 类：处理文本文件解析
  - `parse(file_path)` 方法：解析文本文件并返回结构化数据

示例用法：

```python
from pypile.config_parser import ConfigParser

config_parser = ConfigParser()
config_parser.parse('config.yaml')
data = config_parser.get_data()
config_parser.validate_data()
```

## 运行测试

要运行测试用例，请执行以下命令：

```bash
python -m unittest discover tests
```
