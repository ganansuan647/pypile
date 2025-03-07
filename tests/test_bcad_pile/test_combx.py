#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试COMBX函数 - 该函数用于组合桩基础的刚度矩阵
"""

import unittest
import numpy as np
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


class TestCOMBX(unittest.TestCase):
    """COMBX函数的测试类"""

    @classmethod
    def setUpClass(cls) -> None:
        """
        在所有测试前运行一次，编译Fortran代码
        """
        # 设置F2PY输出文件的路径
        cls.fortran_dir = project_root / "tests" / "test_bcad_pile" / "test_modules"
        fortran_file = cls.fortran_dir / "combx.f"

        # 使用共用的工具函数编译Fortran代码为Python模块
        try:
            cls.combx_fortran = compile_fortran_module(
                fortran_file_path=fortran_file,
                module_name="combx_fortran",
                working_dir=cls.fortran_dir
            )
            print("COMBX函数Fortran代码编译成功")
        except Exception as e:
            print(f"编译COMBX函数Fortran代码时出错: {e}")
            cls.combx_fortran = None

    def test_combx_simple_case(self) -> None:
        """
        测试COMBX函数的简单情况
        """
        # 创建测试数据
        kbx = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
            dtype=np.float32,
        )

        kfr = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5, 1.6],
            ],
            dtype=np.float32,
        )

        # 使用Python实现计算结果
        k1_py = self.py_combx(kbx, kfr)

        # 导入Fortran模块并测试
        try:
            sys.path.insert(
                0, str(project_root / "tests" / "test_bcad_pile" / "test_modules")
            )
            import combx_fortran

            # 创建输出矩阵
            k1_fortran = np.zeros((4, 4), dtype=np.float32)

            # 调用Fortran函数
            combx_fortran.combx(kbx, kfr, k1_fortran)

            # 比较结果
            np.testing.assert_array_almost_equal(k1_py, k1_fortran, decimal=5)

        except ImportError:
            self.skipTest("无法导入编译后的Fortran模块 combx_fortran")

    def py_combx(self, kbx: np.ndarray, kfr: np.ndarray) -> np.ndarray:
        """
        COMBX函数的Python实现，组合桩基础的刚度矩阵

        Args:
            kbx: 4x4刚度矩阵KBX
            kfr: 4x4刚度矩阵KFR

        Returns:
            np.ndarray: 4x4组合后的刚度矩阵K1
        """
        # 初始化结果矩阵
        k1 = np.zeros((4, 4), dtype=np.float32)

        # 组合刚度矩阵
        for i in range(2):
            for j in range(2):
                k1[i, j] = kbx[i, j] + kfr[0, 0]

        k1[0, 1] = k1[0, 1] - kfr[0, 1]
        k1[0, 2] = kbx[0, 2] + kfr[0, 2]
        k1[0, 3] = kbx[0, 3]

        k1[1, 0] = k1[1, 0] - kfr[1, 0]
        k1[1, 1] = k1[1, 1] + kfr[1, 1]
        k1[1, 2] = kbx[1, 2] - kfr[1, 2]
        k1[1, 3] = kbx[1, 3]

        k1[2, 0] = kbx[2, 0] + kfr[2, 0]
        k1[2, 1] = kbx[2, 1] - kfr[2, 1]
        k1[2, 2] = kbx[2, 2] + kfr[2, 2]
        k1[2, 3] = kbx[2, 3]

        k1[3, 0] = kbx[3, 0]
        k1[3, 1] = kbx[3, 1]
        k1[3, 2] = kbx[3, 2]
        k1[3, 3] = kbx[3, 3]

        return k1

    def test_combx_zero_case(self) -> None:
        """
        测试COMBX函数在零矩阵情况下的行为
        """
        # 创建零矩阵测试数据
        kbx = np.zeros((4, 4), dtype=np.float32)
        kfr = np.zeros((4, 4), dtype=np.float32)

        # 使用Python实现计算结果
        k1_py = self.py_combx(kbx, kfr)

        # 验证结果是零矩阵
        expected = np.zeros((4, 4), dtype=np.float32)
        np.testing.assert_array_almost_equal(k1_py, expected, decimal=5)


if __name__ == "__main__":
    unittest.main()
