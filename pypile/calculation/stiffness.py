#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
刚度计算模块
"""
import numpy as np
from typing import Tuple, List, Any


def calculate_area_and_axial_stiffness(pile: Any, pnum: int, zfr: List[float], zbl: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算桩底面积和轴向刚度
    
    Args:
        pile: 桩对象
        pnum: 桩数量
        zfr: 桩地上段高度列表
        zbl: 桩地下段高度列表
        
    Returns:
        tuple: (ao, rzz) 桩底面积和轴向刚度
    """
    print("\n\n       *** To calculate axis stiffness of piles ***")
    ao = np.zeros(pnum, dtype=float)
    pile.area(pnum, zfr, zbl, ao)
    rzz = np.zeros(pnum, dtype=float)
    pile.stiff_n(pnum, zfr, zbl, ao, rzz)
    return ao, rzz


def calculate_lateral_stiffness(pile: Any, pnum: int, rzz: np.ndarray, btx: np.ndarray, bty: np.ndarray) -> None:
    """
    计算桩的侧向刚度
    
    Args:
        pile: 桩对象
        pnum: 桩数量
        rzz: 轴向刚度列表
        btx: x方向变形因子
        bty: y方向变形因子
    """
    print("\n\n       *** To calculate lateral stiffness of piles ***")
    pile.pstiff(pnum, rzz, btx, bty)
