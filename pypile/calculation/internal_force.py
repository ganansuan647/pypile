#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
内力计算模块
"""
from typing import Any, List
import numpy as np


def calculate_pile_displacement_and_force(pile: Any, pnum: int, btx: np.ndarray, bty: np.ndarray, zbl: List[float], duk: np.ndarray) -> None:
    """
    计算桩体的位移和内力
    
    Args:
        pile: 桩对象
        pnum: 桩数量
        btx: x方向变形因子
        bty: y方向变形因子
        zbl: 桩地下段高度列表
        duk: 桩的位移
    """
    # 计算桩体的位移和内力
    pile.eforce(pnum, btx, bty, zbl, duk)
    
