#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模拟桩信息模型
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional
import re

# 地上桩段信息
class SimuAboveGroundSection(BaseModel):
    HFR: float = Field(..., description="地上桩段高度")
    DOF: float = Field(..., description="地上桩段直径/边长")
    NSF: int = Field(..., description="地上桩段的计算单元数")
    
    @field_validator('NSF')
    @classmethod
    def validate_nsf(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("地上桩段计算单元数必须大于0")
        return v

# 地下桩段信息
class SimuBelowGroundSection(BaseModel):
    HBL: float = Field(..., description="地下桩段高度")
    DOB: float = Field(..., description="地下桩段直径/边长")
    PMT: float = Field(..., description="土弹性模量")
    PFI: float = Field(..., description="土内摩擦角(度)")
    NSG: int = Field(..., description="地下桩段的计算单元数")
    
    @field_validator('PFI')
    @classmethod
    def validate_pfi(cls, v: float) -> float:
        if v < 0 or v > 90:
            raise ValueError("土内摩擦角必须在0-90度范围内")
        return v
    
    @field_validator('NSG')
    @classmethod
    def validate_nsg(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("地下桩段计算单元数必须大于0")
        return v

# 单个模拟桩信息模型
class SimuPileModel(BaseModel):
    KSH: int = Field(..., description="桩截面形状控制参数")
    KSU: int = Field(..., description="桩底边界条件控制参数")
    AGL: List[float] = Field(..., description="桩身倾角方向余弦")
    NFR: int = Field(..., description="地面以上桩段数")
    above_ground_sections: List[SimuAboveGroundSection] = Field(..., description="地上桩段信息列表")
    NBL: int = Field(..., description="地面以下桩段数")
    below_ground_sections: List[SimuBelowGroundSection] = Field(..., description="地下桩段信息列表")
    PMB: float = Field(..., description="桩底土弹性模量")
    PEH: float = Field(..., description="桩体弹性模量")
    PKE: float = Field(..., description="形状系数")
    
    @field_validator('KSH')
    @classmethod
    def validate_ksh(cls, v: int) -> int:
        if v not in [0, 1]:
            raise ValueError("KSH必须是0(圆截面)或1(方截面)")
        return v
    
    @field_validator('KSU')
    @classmethod
    def validate_ksu(cls, v: int) -> int:
        if v not in [1, 2, 3, 4]:
            raise ValueError("KSU必须是1(摩擦桩), 2(摩擦端承桩), 3(端承桩)或4(其他边界条件)")
        return v
    
    @field_validator('AGL')
    @classmethod
    def validate_agl(cls, agl: List[float]) -> List[float]:
        if len(agl) != 3:
            raise ValueError("AGL必须包含3个方向余弦值")
        
        # 验证方向余弦的平方和是否约等于1
        sum_squares = sum(x**2 for x in agl)
        if abs(sum_squares - 1.0) > 0.001:  # 允许有小误差
            raise ValueError(f"方向余弦的平方和必须等于1，当前值为{sum_squares}")
        
        return agl
    
    @model_validator(mode='after')
    def validate_sections_count(self) -> 'SimuPileModel':
        """验证桩段信息数量是否符合NFR和NBL"""
        if len(self.above_ground_sections) != self.NFR:
            raise ValueError(f"地上桩段信息数量({len(self.above_ground_sections)})与NFR({self.NFR})不符")
            
        if len(self.below_ground_sections) != self.NBL:
            raise ValueError(f"地下桩段信息数量({len(self.below_ground_sections)})与NBL({self.NBL})不符")
            
        return self

# 模拟桩分组信息模型
class SimuPileModel(BaseModel):
    KSCTR: List[int] = Field(..., description="模拟桩类型标识")
    pile_types: Dict[int, SimuPileModel] = Field(..., description="模拟桩类型信息")
    
    @model_validator(mode='after')
    def validate_ksctr_keys(self) -> 'SimuPileModel':
        """验证KSCTR中的所有值都在pile_types字典中有对应的键"""
        for k in self.KSCTR:
            if k not in self.pile_types:
                raise ValueError(f"KSCTR({self.model_fields['KSCTR'].description})中存在值:{k},但并未找到对应的土层信息，请检查!")
                
        return self

# 模拟桩信息解析模型
class SimuPileInfoModel(BaseModel):
    simu_pile: SimuPileModel
    
    @model_validator(mode='before')
    @classmethod
    def parse_input(cls, data: Any) -> Dict[str, Any]:
        # 解析输入文本
        raw_input = data.get('input_text', '')
        if not raw_input:
            raise ValueError("输入不能为空")
        
        # 使用不区分大小写的模式检查是否存在桩信息标签
        # 只支持[SIMUPILE], [Simu pile], [simu_pe]等具体形式
        tag_pattern = re.compile(r'\[(simu[\s_]*(pile|pe))\]', re.IGNORECASE)
        if not tag_pattern.search(raw_input):
            raise ValueError("无法找到有效的模拟桩信息标签[Simu Pile]或[simu_pe]")
        
        if 'END' not in raw_input and 'end' not in raw_input:
            raw_input += '\nEND;'
            
        # 将多个\n替换为\n
        raw_input = re.sub(r'\n+', '\n', raw_input)
        
        # 解析指定形式的桩信息标签到END;或end;之间的内容
        simu_pile_pattern = r'\[(simu[\s_]*(pile|pe))\].*?(?:END;|end;)'
        match = re.search(simu_pile_pattern, raw_input, re.DOTALL | re.IGNORECASE)
        if match:
            input_text = match.group(0)
        else:
            raise ValueError("无法找到有效的模拟桩信息块")
        
        # 清理多余的空格和换行（仅支持指定形式的标签）
        content = re.sub(r'\[(simu[\s_]*(pile|pe))\]|\s*(?:END;|end;)', '', input_text, flags=re.DOTALL | re.IGNORECASE).strip()
        
        # 按行分割内容
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return {'simu_pile': SimuPileModel(
            KSCTR=[],
            pile_types={}
        )}
        
        # 解析KSCTR列表
        ksctr_values = list(map(int, re.findall(r'-?\d+', lines[0])))
        
        # 解析每种桩类型的信息
        pile_types = {}
        
        # 查找所有桩类型标识段
        type_markers = []
        for i, line in enumerate(lines[1:], 1):
            if re.match(r'<-?\d+>', line):
                type_markers.append((i, int(re.search(r'<(-?\d+)>', line).group(1))))
        
        # 解析每种桩类型的详细信息
        for i, (start_line, pile_type) in enumerate(type_markers):
            end_line = type_markers[i+1][0] if i < len(type_markers) - 1 else len(lines)
            pile_data = "\n".join(lines[start_line+1:end_line])
            
            # 提取所有数值
            values = list(map(float, re.findall(r'-?\d+\.?\d*', pile_data)))
            if len(values) < 12:  # 至少需要基本字段数据
                raise ValueError(f"桩类型{pile_type}的信息不完整")
            
            # 解析基本字段
            ksh = int(values[0])
            ksu = int(values[1])
            agl = values[2:5]
            
            # 解析地上桩段
            nfr = int(values[5])
            idx = 6
            above_ground_sections = []
            for _ in range(nfr):
                if idx + 2 >= len(values):
                    raise ValueError(f"桩类型{pile_type}的地上桩段信息不完整")
                above_ground_sections.append(SimuAboveGroundSection(
                    HFR=values[idx],
                    DOF=values[idx+1],
                    NSF=int(values[idx+2])
                ))
                idx += 3
            
            # 解析地下桩段
            nbl = int(values[idx])
            idx += 1
            below_ground_sections = []
            for _ in range(nbl):
                if idx + 4 >= len(values):
                    raise ValueError(f"桩类型{pile_type}的地下桩段信息不完整")
                below_ground_sections.append(SimuBelowGroundSection(
                    HBL=values[idx],
                    DOB=values[idx+1],
                    PMT=values[idx+2],
                    PFI=values[idx+3],
                    NSG=int(values[idx+4])
                ))
                idx += 5
            
            # 解析剩余字段
            if idx + 2 >= len(values):
                raise ValueError(f"桩类型{pile_type}的信息不完整，缺少PMB{pile_type.model_fields['PMB'].description}、PEH{pile_type.model_fields['PEH'].description}或PKE{pile_type.model_fields['PKE'].description}")
            pmb = values[idx]
            peh = values[idx+1]
            pke = values[idx+2]
            
            # 构建桩类型模型
            pile_types[pile_type] = SimuPileModel(
                KSH=ksh,
                KSU=ksu,
                AGL=agl,
                NFR=nfr,
                above_ground_sections=above_ground_sections,
                NBL=nbl,
                below_ground_sections=below_ground_sections,
                PMB=pmb,
                PEH=peh,
                PKE=pke
            )
        
        # 构建SimuPileGroup
        data['simu_pile'] = SimuPileModel(
            KSCTR=ksctr_values,
            pile_types=pile_types
        )
        
        return data

# 工厂函数：根据输入文本创建适当的模型实例
def parse_simu_pile_text(input_text: str) -> SimuPileInfoModel:
    return SimuPileInfoModel(input_text=input_text)


if __name__ == "__main__":
    # 测试输入
    # input_text = """
    # [simu_pe]
    # 4 5
    # <4>
    # 0 2 0.0 0.0 1.0
    # 1 5.0 1.0 2
    # 2 10.0 1.0 30000.0 30.0 3 15.0 1.0 28000.0 28.0 3
    # 25000.0 30000000.0 1.0
    # <5>
    # 1 3 0.0 0.0 1.0
    # 0
    # 1 10.0 1.0 35000.0 30.0 2
    # 30000.0 32000000.0 1.0
    # END;
    # """
    input_text = """
    [simu_pe]
    END;
    """
    
    model = parse_simu_pile_text(input_text)
    print(f"KSCTR({model.simu_pile.model_fields['KSCTR'].description}): {model.simu_pile.KSCTR}")
    print(f"桩类型数量: {len(model.simu_pile.pile_types)}")
    for type_id, pile_type in model.simu_pile.pile_types.items():
        print(f"桩类型 {type_id}:")
        print(f"  KSH({pile_type.model_fields['KSH'].description}): {pile_type.KSH}, KSU({pile_type.model_fields['KSU'].description}): {pile_type.KSU}")
        print(f"  地上桩段数: {pile_type.NFR}")
        print(f"  地下桩段数: {pile_type.NBL}")
