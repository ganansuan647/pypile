#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试CNDTN函数 - 该函数用于处理边界条件
"""

import unittest
import numpy as np
import glob
from typing import List, Dict, Tuple, Optional, Union, Any
import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入Fortran编译工具
from tests.fortran_utils import compile_fortran_module


class TestCNDTN(unittest.TestCase):
    """CNDTN函数的测试类"""

    @classmethod
    def setUpClass(cls) -> None:
        """
        在所有测试前运行一次，编译Fortran代码
        """
        # 设置F2PY输出文件的路径
        cls.fortran_dir = project_root / "tests" / "test_bcad_pile" / "test_modules"
        fortran_file = cls.fortran_dir / "cndtn.f"

        # 使用共用的工具函数编译Fortran代码为Python模块
        try:
            cls.cndtn_fortran = compile_fortran_module(
                fortran_file_path=fortran_file,
                module_name="cndtn_fortran",
                working_dir=cls.fortran_dir
            )
            print("CNDTN函数Fortran代码编译成功")

        except Exception as e:
            print(f"编译CNDTN函数Fortran代码时出错: {e}")
            cls.cndtn_fortran = None

        # 打印导入的模块信息，帮助调试
        if cls.cndtn_fortran:
            print(f"模块属性: {dir(cls.cndtn_fortran)}")

    def test_cndtn_ksu_1(self) -> None:
        """
        测试CNDTN函数在KSU=1(钻孔灌注桩)情况下的行为
        """
        if not hasattr(self, "cndtn_fortran") or not self.cndtn_fortran:
            self.skipTest("Fortran模块未成功加载")

        # 创建测试数据
        ksu = 1  # 钻孔灌注桩

        # 创建测试刚度矩阵
        kx = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            dtype=np.float32,
        )

        ky = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5, 1.6],
            ],
            dtype=np.float32,
        )

        rzz = 100.0  # 轴向刚度

        # 使用Python实现计算结果
        ke_py = self.py_cndtn(ksu, kx, ky, rzz)

        # 创建输出矩阵
        ke_fortran = np.zeros((6, 6), dtype=np.float32)

        try:
            # 调用Fortran函数
            self.cndtn_fortran.cndtn(ksu, kx, ky, rzz, ke_fortran)

            # 比较结果
            print("\n比较Python和Fortran实现的结果 (KSU=1):")
            for i in range(6):
                for j in range(6):
                    print(
                        f"KE[{i + 1},{j + 1}] - Python: {ke_py[i, j]:.6f}, Fortran: {ke_fortran[i, j]:.6f}, "
                        f"差异: {abs(ke_py[i, j] - ke_fortran[i, j]):.6f}"
                    )

            np.testing.assert_array_almost_equal(ke_py, ke_fortran, decimal=5)

        except Exception as e:
            self.fail(f"Fortran函数调用失败: {e}")

    def test_cndtn_ksu_2(self) -> None:
        """
        测试CNDTN函数在KSU=2(打入摩擦桩)情况下的行为
        """
        if not hasattr(self, "cndtn_fortran") or not self.cndtn_fortran:
            self.skipTest("Fortran模块未成功加载")

        # 创建测试数据
        ksu = 2  # 打入摩擦桩

        # 创建测试刚度矩阵
        kx = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            dtype=np.float32,
        )

        ky = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5, 1.6],
            ],
            dtype=np.float32,
        )

        rzz = 100.0  # 轴向刚度

        # 使用Python实现计算结果
        ke_py = self.py_cndtn(ksu, kx, ky, rzz)

        # 创建输出矩阵
        ke_fortran = np.zeros((6, 6), dtype=np.float32)

        try:
            # 调用Fortran函数
            self.cndtn_fortran.cndtn(ksu, kx, ky, rzz, ke_fortran)

            # 比较结果
            print("\n比较Python和Fortran实现的结果 (KSU=2):")
            for i in range(6):
                for j in range(6):
                    print(
                        f"KE[{i + 1},{j + 1}] - Python: {ke_py[i, j]:.6f}, Fortran: {ke_fortran[i, j]:.6f}, "
                        f"差异: {abs(ke_py[i, j] - ke_fortran[i, j]):.6f}"
                    )

            np.testing.assert_array_almost_equal(ke_py, ke_fortran, decimal=5)

        except Exception as e:
            self.fail(f"Fortran函数调用失败: {e}")

    def test_cndtn_ksu_4(self) -> None:
        """
        测试CNDTN函数在KSU=4(端承嵌固桩)情况下的行为
        """
        if not hasattr(self, "cndtn_fortran") or not self.cndtn_fortran:
            self.skipTest("Fortran模块未成功加载")

        # 创建测试数据
        ksu = 4  # 端承嵌固桩

        # 创建测试刚度矩阵
        kx = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            dtype=np.float32,
        )

        ky = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5, 1.6],
            ],
            dtype=np.float32,
        )

        rzz = 100.0  # 轴向刚度

        # 使用Python实现计算结果
        ke_py = self.py_cndtn(ksu, kx, ky, rzz)

        # 创建输出矩阵
        ke_fortran = np.zeros((6, 6), dtype=np.float32)

        try:
            # 调用Fortran函数
            self.cndtn_fortran.cndtn(ksu, kx, ky, rzz, ke_fortran)

            # 比较结果
            print("\n比较Python和Fortran实现的结果 (KSU=4):")
            for i in range(6):
                for j in range(6):
                    print(
                        f"KE[{i + 1},{j + 1}] - Python: {ke_py[i, j]:.6f}, Fortran: {ke_fortran[i, j]:.6f}, "
                        f"差异: {abs(ke_py[i, j] - ke_fortran[i, j]):.6f}"
                    )

            # 验证KSU=4时轴向刚度应该乘以100
            self.assertAlmostEqual(ke_fortran[2, 2], 100.0 * rzz, delta=0.001)

            np.testing.assert_array_almost_equal(ke_py, ke_fortran, decimal=5)

        except Exception as e:
            self.fail(f"Fortran函数调用失败: {e}")

    def py_cndtn(
        self, ksu: int, kx: np.ndarray, ky: np.ndarray, rzz: float
    ) -> np.ndarray:
        """
        CNDTN函数的Python实现，处理边界条件

        Args:
            ksu: 桩的支撑类型 (1:钻孔灌注桩, 2:打入摩擦桩, 3:端承非嵌固桩, 4:端承嵌固桩)
            kx: 4x4刚度矩阵KX
            ky: 4x4刚度矩阵KY
            rzz: 轴向刚度

        Returns:
            np.ndarray: 6x6组合后的刚度矩阵KE
        """
        # 初始化结果矩阵
        ke = np.zeros((6, 6), dtype=np.float32)

        # 设置刚度矩阵元素
        ke[0, 0] = kx[1, 1]
        ke[0, 4] = -kx[1, 2]
        ke[1, 1] = ky[1, 1]
        ke[1, 3] = ky[1, 2]
        ke[2, 2] = rzz
        ke[3, 1] = ky[2, 1]
        ke[3, 3] = ky[2, 2]
        ke[4, 0] = -kx[2, 1]
        ke[4, 4] = kx[2, 2]
        ke[5, 5] = kx[3, 3] + ky[3, 3]

        # 根据支撑类型设置其他元素
        if ksu == 2:
            ke[1, 5] = ky[1, 3] + ke[1, 4]
            ke[3, 5] = ky[2, 3] + ke[3, 4]
            ke[5, 1] = ky[3, 1] + ke[5, 0]
            ke[5, 3] = ky[3, 2] + ke[5, 2]
        else:
            ke[0, 5] = kx[1, 3]
            ke[1, 5] = ky[1, 3]
            ke[3, 5] = ky[2, 3]
            ke[4, 5] = kx[2, 3]
            ke[5, 0] = kx[3, 1]
            ke[5, 1] = ky[3, 1]
            ke[5, 3] = ky[3, 2]
            ke[5, 4] = kx[3, 2]

        # 补充下三角部分，确保对称性
        for i in range(5):
            for j in range(i + 1, 6):
                ke[j, i] = ke[i, j]

        # 对于KSU=4（端承嵌固桩），轴向刚度乘以100
        if ksu == 4:
            ke[2, 2] = 100.0 * ke[2, 2]

        return ke


if __name__ == "__main__":
    unittest.main()
