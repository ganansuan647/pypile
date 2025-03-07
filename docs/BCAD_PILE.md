# BCAD_PILE 桥梁桩基分析程序文档

## 1. 程序概述

BCAD_PILE 是一个用FORTRAN语言开发的工程分析软件，专门用于桥梁子结构桩基础的空间静力分析。该程序可以计算单桩和桩组的位移、内力以及土壤应力等关键参数，为桥梁设计提供重要依据。

### 1.1 程序功能

- 读取桩基础构型及材料参数
- 计算桩的变形因子
- 计算桩底面积
- 计算桩的轴向刚度
- 计算桩的横向刚度
- 分析整个桩基础系统并计算桩帽中心的位移
- 计算每根桩身的位移和内力分布

### 1.2 程序适用范围

- 圆形和方形截面桩
- 不同边界条件的桩基础
- 考虑桩间距效应的桩组分析
- 不同土层的桩-土相互作用分析

## 2. 输入文件格式

BCAD_PILE程序需要一个扩展名为`.dat`的输入文件，该文件包含以下几个主要数据块：

### 2.1 [CONTRAL] 控制信息

```
[CONTRAL]
JCTR
```

其中：
- JCTR为控制参数
  - JCTR = 1：执行完整分析
  - JCTR = 2：仅计算整个桩基础的刚度矩阵
  - JCTR = 3：仅计算指定单桩的刚度矩阵，需要额外读取指定桩号INO

当JCTR = 1时，还需要输入外力信息：
```
NACT
X1 Y1
FX1 FY1 FZ1 MX1 MY1 MZ1
X2 Y2
FX2 FY2 FZ2 MX2 MY2 MZ2
...
```

### 2.2 [ARRANGE] 桩布置信息

```
[ARRANGE]
PNUM SNUM
X1 Y1
X2 Y2
...
XS1 YS1
XS2 YS2
...
```

其中：
- PNUM：非模拟桩的数量
- SNUM：模拟桩的数量
- Xi, Yi：第i个非模拟桩的平面坐标
- XSi, YSi：第i个模拟桩的平面坐标

### 2.3 [NO_SIMU] 非模拟桩信息

```
[NO_SIMU]
KCTR(1) KCTR(2) ... KCTR(PNUM)
<0>
KSH1 KSU1 AGL1(1) AGL1(2) AGL1(3)
NFR1 HFR1(1) DOF1(1) NSF1(1) ... HFR1(NFR1) DOF1(NFR1) NSF1(NFR1)
NBL1 HBL1(1) DOB1(1) PMT1(1) PFI1(1) NSG1(1) ... HBL1(NBL1) DOB1(NBL1) PMT1(NBL1) PFI1(NBL1) NSG1(NBL1)
PMB1 PEH1 PKE1
```

对于KCTR数组中的每一个不同取值IM(包括负值和正值)，都需要定义一组<IM>段信息：

```
<IM>
...
```

### 2.4 [SIMUPILE] 模拟桩信息

```
[SIMUPILE]
KSCTR(1) KSCTR(2) ... KSCTR(SNUM)
<IM1>
...
<IM2>
...
```

## 3. 参数说明

### 3.1 基本几何与物理参数

- **KSH**：桩截面形状控制参数
  - KSH = 0：圆截面
  - KSH = 1：方截面

- **KSU**：桩底边界条件控制参数
  - KSU = 1：摩擦桩
  - KSU = 2：摩擦端承桩
  - KSU = 3：端承桩
  - KSU = 4：其他边界条件

- **AGL**：桩身倾角方向余弦
  - AGL(1)：X方向余弦
  - AGL(2)：Y方向余弦
  - AGL(3)：Z方向余弦

- **NFR**：地面以上桩段数
- **HFR**：各地上桩段高度
- **DOF**：各地上桩段直径/边长
- **NSF**：各地上桩段的计算单元数

- **NBL**：地面以下桩段数
- **HBL**：各地下桩段高度
- **DOB**：各地下桩段直径/边长
- **PMT**：土弹性模量
- **PFI**：土内摩擦角(度)
- **NSG**：各地下桩段的计算单元数

- **PMB**：桩底土弹性模量
- **PEH**：桩体弹性模量
- **PKE**：形状系数

### 3.2 计算结果参数

- **UX, UY, UZ**：X, Y, Z方向位移(m)
- **SX, SY, SZ**：绕X, Y, Z轴的转角(rad)
- **NX, NY, NZ**：X, Y, Z方向内力(t)
- **MX, MY, MZ**：绕X, Y, Z轴的弯矩(t*m)
- **PSX, PSY**：X, Y方向土压应力(t/m²)

## 4. 输出文件说明

程序生成两个输出文件：
- `.out`：包含详细计算结果的文本文件
- `.pos`：包含后处理数据的文件

### 4.1 .out文件内容

输出文件包含以下主要部分：
1. 程序标题和版权信息
2. 计算的桩基础刚度矩阵(当JCTR=2)
3. 计算的单桩刚度矩阵(当JCTR=3)
4. 桩帽中心的位移和转角(当JCTR=1)
5. 各桩顶部的位移和内力(当JCTR=1)
6. 各桩身的位移和土压应力分布(当JCTR=1)
7. 各桩身的内力分布(当JCTR=1)

### 4.2 .pos文件内容

`.pos`文件包含用于后处理的数据：
1. 桩的平面位置坐标
2. 各桩的深度坐标
3. 各桩各点的位移和内力数据
4. 各桩各点的土压应力数据

## 5. 计算原理

### 5.1 基本理论

BCAD_PILE程序基于弹性地基梁理论和桩-土相互作用模型，主要计算步骤如下：

1. 考虑桩间距影响因子(KINF)计算桩的变形系数
2. 对每根桩计算轴向刚度和横向刚度
3. 将各桩刚度组装成整个基础系统的刚度矩阵
4. 求解位移方程获得桩帽的位移和转角
5. 计算各桩的内力分布和土压应力

### 5.2 关键算法

#### 5.2.1 桩变形系数计算

程序中通过BTX和BTY参数表示桩在X和Y方向的变形系数，计算公式为：
```
BTX = (PMT*BX1/(PEH*B))**0.2
BTY = (PMT*BY1/(PEH*B))**0.2
```

其中BX1和BY1与桩间距影响因子KINF有关。

#### 5.2.2 桩-土相互作用矩阵

程序通过SAA子程序计算桩的关系矩阵，考虑了不同深度土层的影响。

#### 5.2.3 桩基础整体刚度矩阵

通过坐标变换将各桩的局部刚度矩阵转换到全局坐标系，并进行组装。

## 6. 使用示例

### 6.1 输入文件示例

```
[CONTRAL]
1
2
0.0 0.0
100.0 0.0 -500.0 0.0 0.0 0.0
5.0 0.0
0.0 0.0 0.0 100.0 0.0 0.0
[END]

[ARRANGE]
4 0
-5.0 -5.0
5.0 -5.0
5.0 5.0
-5.0 5.0
[END]

[NO_SIMU]
0 0 0 0
<0>
0 1 0.0 0.0 1.0
1 5.0 1.2 10
2 15.0 1.2 10 15.0 1.2 2000.0 30.0 10
2000.0 3.0E6 0.8
[END]

[SIMUPILE]
[END]
```

### 6.2 运行程序

编译程序：
```
gfortran -o bcad_pile Pile_guan.f
```

运行程序：
```
./bcad_pile
```
程序将提示输入数据文件名，输入不含扩展名的文件名即可。

### 6.3 结果解读

程序会输出桩帽中心的位移和转角，以及各桩的位移、内力分布和土压应力。这些数据可以用于评估桩基础的稳定性和安全性。

## 7. 程序限制

1. 当前版本最多支持1000个非模拟桩和20个模拟桩
2. 每根桩最多支持15个地上段和15个地下段
3. 桩的变形系数计算模型基于经验公式，适用于常规土质条件
4. 未考虑动力荷载和非线性土体行为

## 8. 程序文件说明

### 8.1 源文件

- `Pile_guan.f`：原始版本，支持最多200个非模拟桩
- `Pile-w.f`：修改版本，支持最多1000个非模拟桩，由王志强修改

### 8.2 子程序功能说明

- `R_DATA`：读取初始结构数据
- `INIT1`, `INIT2`, `INIT3`, `INIT4`, `INIT5`, `INIT6`：初始化相关子程序
- `BTXY`：计算桩的变形因子
- `KINF1`, `KINF2`, `KINF3`：计算影响因子
- `AREA`：计算桩底面积
- `STN`：计算单桩轴向刚度
- `STIFF_N`：计算各桩轴向刚度
- `PSTIFF`：计算桩的横向刚度
- `DISP`：计算桩帽位移
- `EFORCE`：计算桩身位移和内力

## 9. 版权信息

BCAD_PILE程序版权归同济大学桥梁工程系CAD研究组所有，版本号1.10。

## 10. 联系方式

如有任何关于BCAD_PILE程序的问题，请联系：

CAD研究组  
同济大学桥梁工程系  
上海四平路1239号  
邮编：200092  
中国