#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
位移计算模块
"""
import numpy as np
from typing import Tuple, Any


def calculate_cap_displacement(pile: Any, jctr: int, ino: int, pnum: int, snum: int, force: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算桩基础帽的位移
    
    Args:
        pile: 桩对象
        jctr: 控制参数
        ino: 输出控制参数
        pnum: 桩数量
        snum: 模拟桩数量
        force: 外部荷载数组
        
    Returns:
        tuple: (duk, so) 桩的位移和刚度矩阵
    """
    print("\n\n       *** To execute entire pile foundation analysis ***\n\n")
    duk = np.zeros((pnum, 6), dtype=float)
    so = np.zeros((6, 6), dtype=float)
    pile.disp(jctr, ino, pnum, snum, force, duk, so)
    return duk, so
