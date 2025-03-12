#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
桩体计算结果模型
"""

from pydantic import BaseModel, Field
from typing import List, Dict


# 桩顶位移和内力模型
class PileTopResult(BaseModel):
    # 位移
    UX: float = Field(..., description="X方向位移(m)")
    UY: float = Field(..., description="Y方向位移(m)")
    UZ: float = Field(..., description="Z方向位移(m)")
    SX: float = Field(..., description="X轴旋转角(rad)")
    SY: float = Field(..., description="Y轴旋转角(rad)")
    SZ: float = Field(..., description="Z轴旋转角(rad)")
    
    # 内力
    NX: float = Field(..., description="X方向轴力(kN)")
    NY: float = Field(..., description="Y方向轴力(kN)")
    NZ: float = Field(..., description="Z方向轴力(kN)")
    MX: float = Field(..., description="X方向弯矩(kN*m)")
    MY: float = Field(..., description="Y方向弯矩(kN*m)")
    MZ: float = Field(..., description="Z方向弯矩(kN*m)")


# 桩体节点结果模型
class PileNodeResult(BaseModel):
    Z: float = Field(..., description="Z坐标(m)")
    D: float = Field(..., description="桩的直径(m)")
    
    # 位移
    UX: float = Field(..., description="X方向位移(m)")
    UY: float = Field(..., description="Y方向位移(m)")
    SX: float = Field(..., description="X轴旋转角(rad)")
    SY: float = Field(..., description="Y轴旋转角(rad)")
    
    # 内力
    NX: float = Field(..., description="X方向轴力(kN)")
    NY: float = Field(..., description="Y方向轴力(kN)")
    NZ: float = Field(..., description="Z方向轴力(kN)")
    MX: float = Field(..., description="X方向弯矩(kN*m)")
    MY: float = Field(..., description="Y方向弯矩(kN*m)")
    
    # 土压力
    PSX: float = Field(..., description="X方向土压力(kN/m2)")
    PSY: float = Field(..., description="Y方向土压力(kN/m2)")


# 桩体计算结果模型
class PileResult(BaseModel):
    pile_number: int = Field(..., description="桩号")
    coordinate: List[float] = Field(..., description="桩顶坐标(x,y)")
    top_result: PileTopResult = Field(..., description="桩顶位移和内力")
    nodes: List[PileNodeResult] = Field(..., description="桩体各节点结果")
    ground_level_index: int = Field(..., description="地面所在节点索引")
