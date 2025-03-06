# PyPile 测试说明

本目录包含用于测试 BCAD_PILE.f 中各个函数的测试文件。这些测试使用 f2py 将 Fortran 代码编译为 Python 模块，然后通过 Python 的 unittest 框架进行测试。

## 测试结构

测试文件组织如下：

```
tests/
├── run_tests.py               # 运行所有测试的主脚本
├── test_bcad_pile/            # BCAD_PILE 相关测试
│   ├── __init__.py            # 测试套件初始化
│   ├── test_kinf3.py          # 测试 KINF3 函数（计算桩的影响因子）
│   ├── test_area.py           # 测试 AREA 函数（计算桩底面积）
│   ├── test_stn.py            # 测试 STN 函数（计算单桩轴向刚度）
│   ├── test_pstiff.py         # 测试 PSTIFF 函数（计算桩的单元刚度）
│   ├── test_disp.py           # 测试 DISP 函数（计算桩基的位移）
│   └── test_modules/          # 包含从 BCAD_PILE.f 提取的单个函数
│       ├── __init__.py        # 模块初始化
│       ├── kinf3.f            # KINF3 函数的 Fortran 代码
│       ├── area.f             # AREA 函数的 Fortran 代码
│       ├── stn.f              # STN 函数的 Fortran 代码
│       ├── pstiff.f           # PSTIFF 函数的 Fortran 代码
│       └── disp.f             # DISP 函数的 Fortran 代码
```

## 测试方法

每个测试文件都遵循相同的模式：

1. 使用 f2py 编译相应的 Fortran 代码
2. 实现一个对应的 Python 版本的函数
3. 使用相同的输入数据调用 Fortran 和 Python 版本
4. 比较两个版本的输出结果

## 运行测试

### 安装依赖

首先确保已安装必要的依赖：

```bash
pip install numpy scipy pytest
```

### 运行所有测试

```bash
python tests/run_tests.py
```

### 运行单个测试

```bash
# 运行特定的测试文件
python -m unittest tests/test_bcad_pile/test_kinf3.py

# 运行特定的测试类
python -m unittest tests.test_bcad_pile.test_kinf3.TestKINF3

# 运行特定的测试方法
python -m unittest tests.test_bcad_pile.test_kinf3.TestKINF3.test_kinf3_single_pile
```

## 测试设计说明

### Python 实现

每个测试文件都包含一个对应 Fortran 函数的 Python 实现。这些 Python 实现遵循以下原则：

1. 使用类型提示增强代码可读性
2. 遵循 Python 最佳实践
3. 尽量与原始 Fortran 代码保持一致的计算逻辑
4. 使用 NumPy 进行高效的数组运算

### 简化测试和完整测试

对于复杂的函数，我们提供了两种测试方法：

1. **简化测试**：测试函数的核心逻辑，不依赖于完整的 COMMON 区域数据
2. **完整测试**：模拟完整的调用环境，包括所有必要的输入数据

## 注意事项

1. 使用 f2py 编译 Fortran 代码需要有适当的 Fortran 编译器（如 gfortran）
2. 某些测试可能需要修改以适应本地环境
3. 某些函数依赖于 BCAD_PILE.f 中的 COMMON 区域，可能需要额外的设置
