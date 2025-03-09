#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
变形因子计算模块
"""
import numpy as np
from typing import Tuple, List, Any


def calculate_deformation_factors(pile: Any, pnum: int, zfr: List[float], zbl: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算桩的变形因子
    
    Args:
        pile: 桩对象
        pnum: 桩数量
        zfr: 桩地上段高度列表
        zbl: 桩地下段高度列表
        
    Returns:
        tuple: (btx, bty) x和y方向的变形因子
    """
    print("\n\n       *** To calculate deformation factors of piles ***")
    btx = np.zeros((pnum, pile.N_max_layer), dtype=float)
    bty = np.zeros((pnum, pile.N_max_layer), dtype=float)
    pile.btxy(pnum, zfr, zbl, btx, bty)
    return btx, bty
