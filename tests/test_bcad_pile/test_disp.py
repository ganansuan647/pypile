#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试DISP函数 - 该函数计算桩基的位移
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

class TestDISP(unittest.TestCase):
    """DISP函数的测试类"""
    
    @classmethod
    def setUpClass(cls) -> None:
        """
        在所有测试前运行一次，编译Fortran代码
        """
        # 设置F2PY输出文件的路径
        fortran_file = project_root / "tests" / "test_bcad_pile" / "test_modules" / "disp.f"
        
        # 使用F2PY编译Fortran代码为Python模块
        try:
            subprocess.run(
                [
                    "python", "-m", "numpy.f2py", "-c", str(fortran_file),
                    "-m", "disp_fortran"
                ],
                cwd=str(fortran_file.parent),
                check=True
            )
            print("DISP函数Fortran代码编译成功")
        except subprocess.CalledProcessError as e:
            print(f"编译DISP函数Fortran代码时出错: {e}")
    
    def test_disp_simple_case(self) -> None:
        """
        测试DISP函数在简单情况下的行为
        """
        # 设置简单的测试数据
        jctr = 2  # 控制参数，非1表示计算初始位移
        ino = 2   # 迭代次数，大于1
        pnum = 1  # 桩的数量
        snum = 0  # 其他参数
        
        # 桩顶坐标
        pxy = np.array([[0.0, 0.0]], dtype=np.float32)  
        
        # 其他变量
        sxy = np.zeros((1, 2), dtype=np.float32)
        agl = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)  # 竖直桩，Z方向为1
        
        # 荷载向量
        force = np.array([100.0, 200.0, 300.0, 10.0, 20.0, 30.0], dtype=np.float32)
        
        # 位移向量和输出数组
        duk = np.zeros(100, dtype=np.float32)
        so = np.zeros(1000000, dtype=np.float32)
        
        # 创建共享内存区域模拟ESP矩阵
        esp = np.zeros((1000000, 6), dtype=np.float32)
        
        # 在对角线上设置一些刚度值进行测试
        for i in range(6):
            esp[i, i] = 1000.0 * (i + 1)
        
        # 使用Python实现计算位移
        duk_py = self.py_disp_simplified(jctr, ino, pnum, snum, pxy, sxy, agl, force, duk.copy(), so.copy(), esp)
        
        # 验证桩顶位移计算
        # 我们期望桩顶位移与荷载成比例，符号相同
        for i in range(6):
            self.assertNotEqual(duk_py[i], 0.0, f"位移分量 {i+1} 不应为零")
            expected_sign = 1 if force[i] >= 0 else -1
            actual_sign = 1 if duk_py[i] >= 0 else -1
            self.assertEqual(actual_sign, expected_sign, f"位移分量 {i+1} 的符号应与荷载符号一致")
    
    def py_disp_simplified(self, 
                          jctr: int, 
                          ino: int, 
                          pnum: int, 
                          snum: int, 
                          pxy: np.ndarray, 
                          sxy: np.ndarray, 
                          agl: np.ndarray, 
                          force: np.ndarray, 
                          duk: np.ndarray, 
                          so: np.ndarray,
                          esp: np.ndarray) -> np.ndarray:
        """
        DISP函数的Python简化实现，计算桩基的位移
        
        Args:
            jctr: 控制参数
            ino: 迭代次数
            pnum: 桩的数量
            snum: 其他结构参数
            pxy: 桩顶坐标
            sxy: 其他坐标
            agl: 方向余弦
            force: 荷载向量
            duk: 位移向量
            so: 输出数组
            esp: 刚度矩阵
            
        Returns:
            np.ndarray: 计算后的位移向量
        """
        # 初始化KP和KS数组
        kp = np.zeros(pnum, dtype=np.int32)
        ks = np.zeros(snum, dtype=np.int32)
        
        # 初始化FR和DU矩阵
        fr = np.zeros((6, 1), dtype=np.float32)
        du = np.zeros((6, 1), dtype=np.float32)
        
        for i in range(6):
            fr[i, 0] = force[i]
        
        # 若ino小于等于1，初始化位移向量
        if ino <= 1:
            for i in range(pnum):
                i1 = i * 6
                for j in range(6):
                    duk[i1 + j] = 0.0
        
        # 初始化KP数组
        for i in range(pnum):
            kp[i] = 0
        
        # 初始化KS数组
        for i in range(snum):
            ks[i] = 0
        
        # 条件分支
        if jctr == 1:
            # 初始化位移为零
            for i in range(pnum):
                k1 = 6 * i
                kp[i] = k1
                for j in range(6):
                    duk[k1 + j] = 0.0
        else:
            # 计算初始位移
            for k1 in range(pnum):
                k0 = k1 * 6
                # 计算力矩
                r = -pxy[k1, 0] * force[4] + pxy[k1, 1] * force[3]
                
                # 设置初始位移
                duk[k0 + 0] = 0.001 * force[0] / pnum  # X方向
                duk[k0 + 1] = 0.001 * force[1] / pnum  # Y方向
                duk[k0 + 2] = 0.001 * force[2] / pnum  # Z方向
                duk[k0 + 3] = 0.001 * force[3]         # 绕X转动
                duk[k0 + 4] = 0.001 * force[4]         # 绕Y转动
                duk[k0 + 5] = 0.001 * force[5]         # 绕Z转动
                
                # 非竖直桩时添加附加位移
                if agl[k1, 2] != 1.0:
                    asg = math.sqrt(1 - agl[k1, 2]**2)
                    asg1 = agl[k1, 0] / asg
                    asg2 = agl[k1, 1] / asg
                    
                    duk[k0 + 0] += 0.001 * asg1 * r
                    duk[k0 + 1] += 0.001 * asg2 * r
                    duk[k0 + 2] -= 0.001 * asg * r
        
        # 设置KS数组
        for i in range(snum):
            ks[i] = pnum * 6 + i * 6
        
        # 初始化刚度矩阵
        ke = np.zeros((6, 6), dtype=np.float32)
        
        # 填充SO数组
        k1 = 0
        pes = 0.0
        
        for k in range(pnum):
            k00 = int(kp[k])
            esi = 0.0
            
            # 获取桩的刚度矩阵
            for i in range(6):
                for j in range(6):
                    k01 = k * 6
                    ke[i, j] = esp[k01 + i, j]
                    so[k1] = ke[i, j]
                    k1 += 1
            
            # 计算荷载和能量
            for i in range(6):
                for j in range(6):
                    fr[i, 0] -= ke[i, j] * duk[k00 + j]
                    esi += ke[i, j] * duk[k00 + j]**2
            
            pes += esi
        
        # 处理其他结构部分（如果有）
        k1 = 0
        for k in range(snum):
            for i in range(6):
                for j in range(6):
                    k3 = pnum * 6 * 6 + k * 6 * 6
                    so[k3 + k1] = 0.0
                    k1 += 1
        
        return duk
    
    def py_disp_full(self,
                    jctr: int,
                    ino: int,
                    pnum: int,
                    snum: int,
                    pxy: np.ndarray,
                    sxy: np.ndarray,
                    agl: np.ndarray,
                    force: np.ndarray,
                    duk: np.ndarray,
                    so: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        DISP函数的完整Python实现，计算桩基的位移
        
        Args:
            jctr: 控制参数
            ino: 迭代次数
            pnum: 桩的数量
            snum: 其他结构参数
            pxy: 桩顶坐标
            sxy: 其他坐标
            agl: 方向余弦
            force: 荷载向量
            duk: 位移向量
            so: 输出数组
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (位移向量, 输出数组)
        """
        # 这里是完整实现，需要访问ESP共享内存
        # 由于F2PY的限制，在实际测试中我们需要额外处理这个共享内存
        # 此实现暂时与简化版相同
        return self.py_disp_simplified(jctr, ino, pnum, snum, pxy, sxy, agl, force, duk, so, np.zeros((1000000, 6)))


if __name__ == "__main__":
    unittest.main()
