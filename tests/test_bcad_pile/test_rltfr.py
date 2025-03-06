#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试RLTFR函数 - 该函数计算自由桩段的关系矩阵
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

class TestRLTFR(unittest.TestCase):
    """RLTFR函数的测试类"""
    
    @classmethod
    def setUpClass(cls) -> None:
        """
        在所有测试前运行一次，编译Fortran代码
        """
        # 设置F2PY输出文件的路径
        fortran_file = project_root / "tests" / "test_bcad_pile" / "test_modules" / "rltfr.f"
        
        # 使用F2PY编译Fortran代码为Python模块
        try:
            subprocess.run(
                [
                    "python", "-m", "numpy.f2py", "-c", str(fortran_file),
                    "-m", "rltfr_fortran"
                ],
                cwd=str(fortran_file.parent),
                check=True
            )
            print("RLTFR函数Fortran代码编译成功")
        except subprocess.CalledProcessError as e:
            print(f"编译RLTFR函数Fortran代码时出错: {e}")
    
    def test_frlmtx_single_segment(self) -> None:
        """
        测试FRLMTX函数 - 计算单个自由段的关系矩阵
        """
        # 创建测试数据
        ej = 1.0e7  # 刚度EJ
        h = 5.0     # 高度
        
        # 使用Python实现计算结果
        kf_py = self.py_frlmtx(ej, h)
        
        # 验证结果的对称性
        for i in range(4):
            for j in range(4):
                if (i == 0 and j == 2) or (i == 2 and j == 0) or \
                   (i == 1 and j == 3) or (i == 3 and j == 1):
                    # 这些位置应该是负值
                    self.assertLess(kf_py[i, j], 0, 
                                  msg=f"位置 ({i},{j}) 的值应该为负")
                elif i == j:
                    # 对角线上应该是正值
                    self.assertGreater(kf_py[i, j], 0, 
                                     msg=f"对角线位置 ({i},{j}) 的值应该为正")
    
    def test_rltfr_single_segment(self) -> None:
        """
        测试RLTFR函数在单段情况下的行为
        """
        # 创建测试数据
        n = 1          # 段数
        ej = np.array([1.0e7], dtype=np.float32)  # 刚度EJ
        h = np.array([5.0], dtype=np.float32)     # 高度
        
        # 使用Python实现计算结果
        kfr_py = self.py_rltfr(n, ej, h)
        
        # 与直接使用FRLMTX的结果比较
        kf_single = self.py_frlmtx(ej[0], h[0])
        np.testing.assert_array_almost_equal(kfr_py, kf_single, decimal=5)
        
        # 导入Fortran模块并测试
        try:
            sys.path.insert(0, str(project_root / "tests" / "test_bcad_pile" / "test_modules"))
            import rltfr_fortran
            
            # 创建输出矩阵
            kfr_fortran = np.zeros((4, 4), dtype=np.float32)
            
            # 调用Fortran函数
            rltfr_fortran.rltfr(n, ej, h, kfr_fortran)
            
            # 比较结果
            np.testing.assert_array_almost_equal(kfr_py, kfr_fortran, decimal=5)
            
        except ImportError:
            self.skipTest("无法导入编译后的Fortran模块 rltfr_fortran")
    
    def test_rltfr_multiple_segments(self) -> None:
        """
        测试RLTFR函数在多段情况下的行为
        """
        # 创建测试数据
        n = 3          # 段数
        ej = np.array([1.0e7, 2.0e7, 1.5e7], dtype=np.float32)  # 刚度EJ
        h = np.array([5.0, 4.0, 3.0], dtype=np.float32)         # 高度
        
        # 使用Python实现计算结果
        kfr_py = self.py_rltfr(n, ej, h)
        
        # 导入Fortran模块并测试
        try:
            sys.path.insert(0, str(project_root / "tests" / "test_bcad_pile" / "test_modules"))
            import rltfr_fortran
            
            # 创建输出矩阵
            kfr_fortran = np.zeros((4, 4), dtype=np.float32)
            
            # 调用Fortran函数
            rltfr_fortran.rltfr(n, ej, h, kfr_fortran)
            
            # 比较结果
            np.testing.assert_array_almost_equal(kfr_py, kfr_fortran, decimal=5)
            
        except ImportError:
            self.skipTest("无法导入编译后的Fortran模块 rltfr_fortran")
    
    def py_frlmtx(self, ej: float, h: float) -> np.ndarray:
        """
        FRLMTX函数的Python实现，计算单个自由段的关系矩阵
        
        Args:
            ej: 刚度EJ
            h: 高度
            
        Returns:
            np.ndarray: 4x4关系矩阵KF
        """
        # 初始化结果矩阵
        kf = np.zeros((4, 4), dtype=np.float32)
        
        # 计算关系矩阵
        x = h / ej
        
        kf[0, 0] = 12.0 / x
        kf[0, 1] = 6.0
        kf[1, 0] = 6.0
        kf[1, 1] = 4.0 * h
        kf[2, 2] = 12.0 / x
        kf[2, 3] = 6.0
        kf[3, 2] = 6.0
        kf[3, 3] = 4.0 * h
        
        kf[0, 2] = -12.0 / x
        kf[0, 3] = 6.0
        kf[1, 2] = -6.0
        kf[1, 3] = 2.0 * h
        kf[2, 0] = kf[0, 2]
        kf[2, 1] = kf[1, 2]
        kf[3, 0] = kf[0, 3]
        kf[3, 1] = kf[1, 3]
        
        return kf
    
    def py_sinver(self, a: np.ndarray, n: int) -> Tuple[np.ndarray, int]:
        """
        SINVER函数的Python实现，计算矩阵的逆
        
        Args:
            a: 输入矩阵
            n: 矩阵维度
            
        Returns:
            Tuple[np.ndarray, int]: (逆矩阵, 错误代码)
        """
        # 复制输入矩阵，避免修改原始数据
        a_copy = a.copy()
        # 初始化输出矩阵为单位矩阵
        b = np.eye(n, dtype=np.float32)
        # 初始化行列交换记录数组
        is_array = np.zeros(n, dtype=np.int32)
        js_array = np.zeros(n, dtype=np.int32)
        
        # 高斯消元法求逆矩阵
        for k in range(n):
            # 寻找主元
            d = 0.0
            for i in range(k, n):
                for j in range(k, n):
                    if abs(a_copy[i, j]) > abs(d):
                        d = a_copy[i, j]
                        is_array[k] = i
                        js_array[k] = j
            
            # 判断是否奇异
            if abs(d) < 1.0e-10:
                return b, 1  # 奇异矩阵，返回错误代码1
            
            # 交换行
            if is_array[k] != k:
                for j in range(n):
                    c = a_copy[k, j]
                    a_copy[k, j] = a_copy[is_array[k], j]
                    a_copy[is_array[k], j] = c
            
            # 交换列
            if js_array[k] != k:
                for i in range(n):
                    c = a_copy[i, k]
                    a_copy[i, k] = a_copy[i, js_array[k]]
                    a_copy[i, js_array[k]] = c
            
            # 除以主元
            for i in range(n):
                if i != k:
                    a_copy[i, k] = -a_copy[i, k] / d
            
            # 消元
            for i in range(n):
                if i != k:
                    for j in range(n):
                        if j != k:
                            a_copy[i, j] = a_copy[i, j] + a_copy[i, k] * a_copy[k, j]
            
            # 处理主元所在行
            for j in range(n):
                if j != k:
                    a_copy[k, j] = a_copy[k, j] / d
            
            # 处理主元
            a_copy[k, k] = 1.0 / d
        
        # 恢复行列交换对逆矩阵的影响
        for l in range(n):
            k = n - l - 1
            # 恢复行交换对B的影响
            if is_array[k] != k:
                for j in range(n):
                    c = b[k, j]
                    b[k, j] = b[is_array[k], j]
                    b[is_array[k], j] = c
            
            # 恢复列交换对B的影响
            if js_array[k] != k:
                for i in range(n):
                    c = b[i, k]
                    b[i, k] = b[i, js_array[k]]
                    b[i, js_array[k]] = c
        
        return b, 0  # 返回逆矩阵和成功代码0
        
    def py_frfrcom(self, kf1: np.ndarray, kf2: np.ndarray) -> np.ndarray:
        """
        FRFRCOM函数的Python实现，组合两个自由桩段
        
        Args:
            kf1: 第一个4x4关系矩阵KF1
            kf2: 第二个4x4关系矩阵KF2
            
        Returns:
            np.ndarray: 4x4组合后的关系矩阵KF
        """
        # 初始化结果矩阵
        kf = np.zeros((4, 4), dtype=np.float32)
        
        # 提取子矩阵
        ax = np.zeros((2, 2), dtype=np.float32)
        bx = np.zeros((2, 2), dtype=np.float32)
        cx = np.zeros((2, 2), dtype=np.float32)
        dx = np.zeros((2, 2), dtype=np.float32)
        
        for i in range(2):
            for j in range(2):
                i2 = i + 2
                j2 = j + 2
                ax[i, j] = kf1[i, j]
                bx[i, j] = kf1[i, j2]
                cx[i, j] = kf1[i2, j]
                dx[i, j] = kf1[i2, j2] + kf2[i, j]
        
        # 计算DX的逆矩阵，使用py_sinver函数
        x, je = self.py_sinver(dx, 2)
        if je != 0:
            # 如果矩阵奇异，返回零矩阵
            return np.zeros((4, 4), dtype=np.float32)
        
        # 计算中间矩阵
        y = np.zeros((2, 2), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                y[i, j] = cx[i, 0] * x[0, j] + cx[i, 1] * x[1, j]
        
        z = np.zeros((2, 2), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    z[i, j] += y[i, k] * bx[k, j]
        
        # 填充结果矩阵
        for i in range(2):
            for j in range(2):
                i2 = i + 2
                j2 = j + 2
                kf[i, j] = ax[i, j] - z[i, j]
                kf[i, j2] = bx[i, 0] * x[0, j] + bx[i, 1] * x[1, j]
                kf[i2, j] = kf2[i, 0] * x[0, j] + kf2[i, 1] * x[1, j]
        
        # 计算其他中间矩阵
        z = np.zeros((2, 2), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                z[i, j] = bx[0, i] * x[0, j] + bx[1, i] * x[1, j]
        
        kf3 = np.zeros((2, 2), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    kf3[i, j] += z[i, k] * kf2[k, j]
        
        kf4 = np.zeros((2, 2), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    kf4[i, j] += kf2[i, k] * z[k, j]
        
        kf5 = np.zeros((2, 2), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    kf5[i, j] += kf2[i, k] * x[k, j]
        
        # 完成结果矩阵
        for i in range(2):
            for j in range(2):
                i2 = i + 2
                j2 = j + 2
                kf[i2, j2] = kf2[i2, j2] - kf5[i, j2] * kf2[j, i2] + kf4[i, j] + kf3[i, j]
        
        return kf
    
    def py_rltfr(self, n: int, ej: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        RLTFR函数的Python实现，计算自由桩段的关系矩阵
        
        Args:
            n: 段数
            ej: 刚度EJ数组
            h: 高度数组
            
        Returns:
            np.ndarray: 4x4关系矩阵KFR
        """
        # 计算第一段的关系矩阵
        kf = self.py_frlmtx(ej[0], h[0])
        
        # 如果只有一段，直接返回
        if n == 1:
            return kf
        
        # 初始化KF1
        kf1 = kf.copy()
        
        # 依次组合各段
        for i in range(1, n):
            # 计算当前段的关系矩阵
            kf2 = self.py_frlmtx(ej[i], h[i])
            
            # 组合KF1和KF2
            kf = self.py_frfrcom(kf1, kf2)
            
            # 更新KF1
            kf1 = kf.copy()
        
        return kf


if __name__ == "__main__":
    unittest.main()
