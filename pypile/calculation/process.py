#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
桩基计算流程管理模块
"""
from typing import Any, List
import numpy as np
from .deformation import calculate_deformation_factors
from .stiffness import calculate_area_and_axial_stiffness, calculate_lateral_stiffness
from .displacement import calculate_cap_displacement
from .internal_force import calculate_pile_displacement_and_force


def run_calculation(pile: Any, jctr: int, ino: int, pnum: int, snum: int, force: np.ndarray, zfr: List[float], zbl: List[float]) -> np.ndarray:
    """
    执行完整的桩基计算流程
    
    Args:
        pile: 桩对象
        jctr: 控制参数
        ino: 输出控制参数
        pnum: 桩数量
        snum: 模拟桩数量
        force: 外部荷载数组
        zfr: 桩地上段高度列表
        zbl: 桩地下段高度列表
        
    Returns:
        np.ndarray: 桩基础刚度矩阵
    """
    # 计算桩的变形因子
    btx, bty = calculate_deformation_factors(pile, pnum, zfr, zbl)
    
    # 计算桩底面积和轴向刚度
    ao, rzz = calculate_area_and_axial_stiffness(pile, pnum, zfr, zbl)
    
    # 计算桩的侧向刚度
    calculate_lateral_stiffness(pile, pnum, rzz, btx, bty)
    
    # 计算桩基础帽的位移
    duk, so = calculate_cap_displacement(pile, jctr, ino, pnum, snum, force)
    
    # 计算桩体的位移和内力
    calculate_pile_displacement_and_force(pile, pnum, btx, bty, zbl, duk)
    
    return so  # 返回桩基础刚度矩阵
