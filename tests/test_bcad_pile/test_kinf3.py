#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试KINF3函数 - 该函数计算桩行的影响因子
"""

import unittest
import numpy as np
from typing import List, Tuple, Optional, Union, Any
import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestKINF3(unittest.TestCase):
    """KINF3函数的测试类"""

    @classmethod
    def setUpClass(cls) -> None:
        """
        在所有测试前运行一次，编译Fortran代码
        """
        # 设置Fortran源文件和F2PY输出文件的路径
        fortran_file = (
            project_root / "tests" / "test_bcad_pile" / "test_modules" / "kinf3.f"
        )

        # 确保目标目录存在
        fortran_file.parent.mkdir(parents=True, exist_ok=True)

        # 从主Fortran文件提取KINF3函数并写入单独的文件
        with open(project_root / "docs" / "BCAD_PILE.f", "r") as src_file:
            src_content = src_file.read()

        # 提取KINF3函数代码
        kinf3_code = """c******************************************************
c   Sub to calculate influential factor of a pile row
c******************************************************
      SUBROUTINE KINF3(IN,AA,DD,ZZ,KINF)
      REAL KINF
      DIMENSION AA(IN),DD(IN),ZZ(IN),HO(1000)
      IF(IN.EQ.1) THEN
        KINF=1.0
        GOTO 2200
      END IF
      DO 140 I=1,IN
        HO(I)=3.0*(DD(I)+1.0)
        IF(HO(I).GT.ZZ(I)) HO(I)=ZZ(I)
140   CONTINUE
      LO=100.0
      DO 141 I=1,IN
        DO 141 I1=I+1,IN
          S=ABS(AA(I)-AA(I1))-(DD(I)+DD(I1))/2.0
          IF(S.LT.LO) THEN
            LO=S
            HOO=HO(I)
            IF(HOO.LT.HO(I1)) HOO=HO(I1)
          END IF
141   CONTINUE  
      IF(LO.GE.0.6*HOO) THEN
        KINF=1.0
      ELSE
        CALL PARC(IN,C)
        KINF=C+(1.0-C)*LO/(0.6*HOO)
      END IF
2200  RETURN
      END

c*****************************************************
c     Sub to give the pile group coefficient of Kinf
c*****************************************************
      SUBROUTINE PARC(IN,C)
      IF(IN.EQ.1) C=1           
      IF(IN.EQ.2) C=0.6
      IF(IN.EQ.3) C=0.5
      IF(IN.GE.4) C=0.45       
      RETURN
      END
"""

        # 将提取的代码写入单独的Fortran文件
        with open(fortran_file, "w") as f:
            f.write(kinf3_code)

        # 使用F2PY编译Fortran代码为Python模块
        try:
            subprocess.run(
                [
                    "python",
                    "-m",
                    "numpy.f2py",
                    "-c",
                    str(fortran_file),
                    "-m",
                    "kinf3_fortran",
                ],
                cwd=str(fortran_file.parent),
                check=True,
            )
            print("Fortran代码编译成功")
        except subprocess.CalledProcessError as e:
            print(f"编译Fortran代码时出错: {e}")
            raise

    def test_kinf3_single_pile(self) -> None:
        """
        测试KINF3在单桩情况下的行为
        """
        # 导入编译后的Fortran模块
        try:
            sys.path.insert(
                0, str(project_root / "tests" / "test_bcad_pile" / "test_modules")
            )
            import kinf3_fortran
        except ImportError:
            self.fail("无法导入编译后的Fortran模块 kinf3_fortran")

        # 测试单桩情况（IN=1）
        aa = np.array([10.0], dtype=np.float32)  # 桩位置
        dd = np.array([1.0], dtype=np.float32)  # 桩直径
        zz = np.array([20.0], dtype=np.float32)  # 桩长度
        kinf = np.array(0.0, dtype=np.float32)  # 结果变量

        # 调用Fortran函数
        kinf3_fortran.kinf3(1, aa, dd, zz, kinf)

        # 验证结果: 单桩情况下KINF=1.0
        self.assertAlmostEqual(kinf, 1.0, places=5)

        # 调用对应的Python实现并比较结果
        py_kinf = self.py_kinf3(aa, dd, zz)
        self.assertAlmostEqual(py_kinf, kinf, places=5)

    def test_kinf3_multi_pile(self) -> None:
        """
        测试KINF3在多桩情况下的行为
        """
        # 导入编译后的Fortran模块
        try:
            sys.path.insert(
                0, str(project_root / "tests" / "test_bcad_pile" / "test_modules")
            )
            import kinf3_fortran
        except ImportError:
            self.fail("无法导入编译后的Fortran模块 kinf3_fortran")

        # 测试多桩情况（IN=3）
        aa = np.array([0.0, 3.0, 6.0], dtype=np.float32)  # 桩位置
        dd = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # 桩直径
        zz = np.array([20.0, 20.0, 20.0], dtype=np.float32)  # 桩长度
        kinf = np.array(0.0, dtype=np.float32)  # 结果变量

        # 调用Fortran函数
        kinf3_fortran.kinf3(3, aa, dd, zz, kinf)

        # 调用对应的Python实现并比较结果
        py_kinf = self.py_kinf3(aa, dd, zz)
        self.assertAlmostEqual(py_kinf, kinf, places=5)

    def py_kinf3(self, aa: np.ndarray, dd: np.ndarray, zz: np.ndarray) -> float:
        """
        KINF3函数的Python实现，计算桩行的影响因子

        Args:
            aa: 桩位置数组
            dd: 桩直径数组
            zz: 桩长度数组

        Returns:
            float: 影响因子KINF值
        """
        in_val = len(aa)  # 桩数量

        # 单桩情况直接返回1.0
        if in_val == 1:
            return 1.0

        # 计算每个桩的HO值
        ho = np.zeros(in_val, dtype=np.float32)
        for i in range(in_val):
            ho[i] = 3.0 * (dd[i] + 1.0)
            if ho[i] > zz[i]:
                ho[i] = zz[i]

        # 初始化LO为一个很大的值
        lo = 100.0
        hoo = 0.0

        # 计算最小间距
        for i in range(in_val):
            for i1 in range(i + 1, in_val):
                s = abs(aa[i] - aa[i1]) - (dd[i] + dd[i1]) / 2.0
                if s < lo:
                    lo = s
                    hoo = ho[i]
                    if hoo < ho[i1]:
                        hoo = ho[i1]

        # 根据LO和HOO计算KINF值
        if lo >= 0.6 * hoo:
            return 1.0
        else:
            # 调用PARC函数获取C值
            c = self.py_parc(in_val)
            return c + (1.0 - c) * lo / (0.6 * hoo)

    def py_parc(self, in_val: int) -> float:
        """
        PARC函数的Python实现，确定桩组系数

        Args:
            in_val: 桩的数量

        Returns:
            float: 桩组系数C
        """
        if in_val == 1:
            return 1.0
        elif in_val == 2:
            return 0.6
        elif in_val == 3:
            return 0.5
        else:  # in_val >= 4
            return 0.45


if __name__ == "__main__":
    unittest.main()
