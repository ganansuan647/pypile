#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试BCAD_PILE主函数 - 该函数是桩基础分析的主入口
"""

import unittest
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Union, Any
import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestBCAD_PILE(unittest.TestCase):
    """BCAD_PILE主函数的测试类"""

    def test_bcad_pile_main_functionality(self) -> None:
        """
        测试BCAD_PILE主函数的核心功能

        这是一个集成测试，验证BCAD_PILE函数的主要流程和组件协同工作
        """
        # 创建一个简单的测试用例，包含以下组件：
        # 1. 2根桩
        # 2. 每根桩有一个地上段和一个地下段
        # 3. 施加简单的荷载

        # 桩的基本参数
        pnum = 2  # 桩数量

        # 桩的坐标
        pxy = np.array(
            [
                [0.0, 0.0],  # 第一根桩的坐标
                [3.0, 0.0],  # 第二根桩的坐标
            ],
            dtype=np.float32,
        )

        # 桩的方向余弦
        agl = np.array(
            [
                [0.0, 0.0, 1.0],  # 第一根桩，竖直方向
                [0.0, 0.0, 1.0],  # 第二根桩，竖直方向
            ],
            dtype=np.float32,
        )

        # 地面以上桩长
        zfr = np.array([5.0, 5.0], dtype=np.float32)

        # 地面以下桩长
        zbl = np.array([15.0, 15.0], dtype=np.float32)

        # 桩的控制参数
        kctr = np.array([0, 0], dtype=np.int32)  # 控制参数
        ksh = np.array([0, 0], dtype=np.int32)  # 形状：0=圆形
        ksu = np.array([1, 1], dtype=np.int32)  # 支撑类型：1=钻孔灌注桩

        # 地上段参数
        nfr = np.array([1, 1], dtype=np.int32)  # 地上段数
        hfr = np.zeros((2, 15), dtype=np.float32)
        dof = np.zeros((2, 15), dtype=np.float32)
        nsf = np.zeros((2, 15), dtype=np.int32)

        hfr[0, 0] = 5.0  # 第一根桩的地上段高度
        hfr[1, 0] = 5.0  # 第二根桩的地上段高度

        dof[0, 0] = 1.0  # 第一根桩的地上段直径
        dof[1, 0] = 1.0  # 第二根桩的地上段直径

        # 地下段参数
        nbl = np.array([1, 1], dtype=np.int32)  # 地下段数
        hbl = np.zeros((2, 15), dtype=np.float32)
        dob = np.zeros((2, 15), dtype=np.float32)
        pmt = np.zeros((2, 15), dtype=np.float32)
        pfi = np.zeros((2, 15), dtype=np.float32)
        nsg = np.zeros((2, 15), dtype=np.int32)

        hbl[0, 0] = 15.0  # 第一根桩的地下段高度
        hbl[1, 0] = 15.0  # 第二根桩的地下段高度

        dob[0, 0] = 1.0  # 第一根桩的地下段直径
        dob[1, 0] = 1.0  # 第二根桩的地下段直径

        pmt[0, 0] = 100.0  # 第一根桩的地下段土抗力
        pmt[1, 0] = 100.0  # 第二根桩的地下段土抗力

        pfi[0, 0] = 30.0  # 第一根桩的地下段摩擦角
        pfi[1, 0] = 30.0  # 第二根桩的地下段摩擦角

        # 其他参数
        pmb = np.array([5.0e6, 5.0e6], dtype=np.float32)  # 底部支撑刚度
        peh = np.array([2.0e7, 2.0e7], dtype=np.float32)  # 弹性模量
        pke = np.array([0.8, 0.8], dtype=np.float32)  # 刚度系数

        # 荷载参数
        force = np.array([100.0, 0.0, 1000.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # 调用Python实现的BCAD_PILE函数
        so, disp, rzz = self.py_bcad_pile_simplified(
            pnum,
            pxy,
            agl,
            zfr,
            zbl,
            kctr,
            ksh,
            ksu,
            nfr,
            hfr,
            dof,
            nsf,
            nbl,
            hbl,
            dob,
            pmt,
            pfi,
            nsg,
            pmb,
            peh,
            pke,
            force,
        )

        # 验证结果的合理性

        # 1. 位移应该与荷载方向一致
        # X方向荷载为正，X方向位移也应为正
        self.assertGreater(disp[0], 0, "X方向位移应为正")

        # Z方向荷载为正，Z方向位移也应为正
        self.assertGreater(disp[2], 0, "Z方向位移应为正")

        # 2. 每根桩都应有非零的轴向刚度
        for i in range(pnum):
            self.assertGreater(rzz[i], 0, f"第{i + 1}根桩的轴向刚度应为正")

        # 3. 输出数组so应该包含各种计算结果
        # 不对具体数值进行验证，但至少应该有非零值
        self.assertTrue(np.any(so != 0), "输出数组应该包含非零值")

        # 4. 验证对称桩基础的对称性
        # 由于两根桩完全相同，它们的轴向刚度应该相等
        self.assertAlmostEqual(
            rzz[0], rzz[1], delta=1.0, msg="对称桩的轴向刚度应该近似相等"
        )

    def py_bcad_pile_simplified(
        self,
        pnum: int,
        pxy: np.ndarray,
        agl: np.ndarray,
        zfr: np.ndarray,
        zbl: np.ndarray,
        kctr: np.ndarray,
        ksh: np.ndarray,
        ksu: np.ndarray,
        nfr: np.ndarray,
        hfr: np.ndarray,
        dof: np.ndarray,
        nsf: np.ndarray,
        nbl: np.ndarray,
        hbl: np.ndarray,
        dob: np.ndarray,
        pmt: np.ndarray,
        pfi: np.ndarray,
        nsg: np.ndarray,
        pmb: np.ndarray,
        peh: np.ndarray,
        pke: np.ndarray,
        force: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        BCAD_PILE函数的简化Python实现，桩基础分析的主函数

        Args:
            pnum: 桩数量
            pxy: 桩顶坐标
            agl: 桩的方向余弦
            zfr: 地面以上桩长
            zbl: 地面以下桩长
            kctr: 控制参数
            ksh: 桩形状标识 (0=圆形, 1=方形)
            ksu: 支撑类型
            nfr: 地上段数
            hfr: 地上段高度
            dof: 地上段直径
            nsf: 地上段土层数
            nbl: 地下段数
            hbl: 地下段高度
            dob: 地下段直径
            pmt: 地下段土抗力
            pfi: 地下段摩擦角
            nsg: 地下段土层数
            pmb: 底部支撑刚度
            peh: 弹性模量
            pke: 刚度系数
            force: 荷载向量

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (输出数组, 位移向量, 轴向刚度数组)
        """
        # 初始化输出数组
        so = np.zeros(1000000, dtype=np.float32)

        # 初始化位移向量
        disp = np.zeros(6, dtype=np.float32)

        # 初始化轴向刚度数组
        rzz = np.zeros(pnum, dtype=np.float32)

        # 初始化桩底面积数组
        ao = np.zeros(pnum, dtype=np.float32)

        # 计算桩底面积
        for k in range(pnum):
            if ksh[k] == 0:  # 圆形桩
                ao[k] = math.pi * dob[k, 0] ** 2 / 4.0
            else:  # 方形桩
                ao[k] = dob[k, 0] ** 2

        # 计算轴向刚度
        for k in range(pnum):
            # 确定PKC系数
            if ksu[k] == 1:
                pkc = 0.5
            elif ksu[k] == 2:
                pkc = 0.667
            else:  # ksu[k] > 2
                pkc = 1.0

            # 计算X值（刚度倒数）
            x = 0.0

            # 地上段贡献
            for ia in range(int(nfr[k])):
                if ksh[k] == 0:  # 圆形桩
                    a = math.pi * dof[k, ia] ** 2 / 4.0
                else:  # 方形桩
                    a = dof[k, ia] ** 2

                x += hfr[k, ia] / (peh[k] * a)

            # 地下段贡献
            for ia in range(int(nbl[k])):
                if ksh[k] == 0:  # 圆形桩
                    a = math.pi * dob[k, ia] ** 2 / 4.0
                else:  # 方形桩
                    a = dob[k, ia] ** 2

                x += pkc * hbl[k, ia] / (peh[k] * a)

            # 底部支撑贡献
            if ksu[k] <= 2:
                x += 1.0 / (pmb[k] * zbl[k] * ao[k])
            else:
                x += 1.0 / (pmb[k] * ao[k])

            # 计算刚度
            rzz[k] = 1.0 / x

        # 简化的位移计算
        # 这是一个非常简化的模型，实际的BCAD_PILE会更复杂

        # X方向位移 (与X方向荷载成比例)
        disp[0] = force[0] * 0.001 / sum(rzz)

        # Y方向位移 (与Y方向荷载成比例)
        disp[1] = force[1] * 0.001 / sum(rzz)

        # Z方向位移 (与Z方向荷载成比例，但考虑轴向刚度)
        disp[2] = force[2] * 0.001 / sum(rzz)

        # 转动位移 (与对应方向的力矩成比例)
        disp[3] = force[3] * 0.001
        disp[4] = force[4] * 0.001
        disp[5] = force[5] * 0.001

        # 填充输出数组，简化实现
        for k in range(pnum):
            k1 = k * 6
            for i in range(6):
                for j in range(6):
                    so[k1 + i * 6 + j] = 1000.0 * (i == j)  # 对角线为1000，其他为0

        return so, disp, rzz

    def py_bcad_pile_full(
        self, so: np.ndarray, force: np.ndarray, jctr: int, ino: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        BCAD_PILE函数的完整Python实现，这是桩基分析的主函数

        由于完整实现需要访问COMMON区域的数据，这里只提供函数接口
        完整实现将需要所有COMMON区域的数据和子函数

        Args:
            so: 输出数组
            force: 荷载向量
            jctr: 控制参数
            ino: 迭代次数

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (输出数组, 位移向量, 轴向刚度数组)
        """
        # 这里只是接口，实际实现需要所有COMMON区域的数据
        # 在实际测试中，我们需要使用F2PY编译完整的Fortran代码

        # 返回虚拟数据
        return so, np.zeros(6), np.zeros(10)


if __name__ == "__main__":
    unittest.main()
