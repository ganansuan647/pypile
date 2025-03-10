#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
非模拟桩信息模型
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional
import re

# 地上桩段信息
class AboveGroundSection(BaseModel):
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
class BelowGroundSection(BaseModel):
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

# 单个非模拟桩信息模型
class NoSimuPileModel(BaseModel):
    KSH: int = Field(..., description="桩截面形状控制参数")
    KSU: int = Field(..., description="桩底边界条件控制参数")
    AGL: List[float] = Field(..., description="桩身倾角方向余弦")
    NFR: int = Field(..., description="地面以上桩段数")
    above_ground_sections: List[AboveGroundSection] = Field(..., description="地上桩段信息列表")
    NBL: int = Field(..., description="地面以下桩段数")
    below_ground_sections: List[BelowGroundSection] = Field(..., description="地下桩段信息列表")
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
    def validate_sections_count(self) -> 'NoSimuPileModel':
        """验证桩段信息数量是否符合NFR和NBL"""
        if len(self.above_ground_sections) != self.NFR:
            raise ValueError(f"地上桩段信息数量({len(self.above_ground_sections)})与NFR({self.NFR})不符")
            
        if len(self.below_ground_sections) != self.NBL:
            raise ValueError(f"地下桩段信息数量({len(self.below_ground_sections)})与NBL({self.NBL})不符")
            
        return self

# 非模拟桩分组信息模型
class NoSimuModel(BaseModel):
    KCTR: List[int] = Field(..., description="非模拟桩类型标识")
    pile_types: Dict[int, NoSimuPileModel] = Field(..., description="非模拟桩类型信息")
    
    @model_validator(mode='after')
    def validate_kctr_keys(self) -> 'NoSimuModel':
        """验证KCTR中的所有值都在pile_types字典中有对应的键"""
        for k in self.KCTR:
            if k not in self.pile_types:
                raise ValueError(f"KCTR({self.model_fields['KCTR'].description})中存在值:{k},但并未找到对应的土层信息，请检查!")
                
        return self

# 非模拟桩信息解析模型
class NoSimuInfoModel(BaseModel):
    no_simu: NoSimuModel
    
    @model_validator(mode='before')
    @classmethod
    def parse_input(cls, data: Any) -> Dict[str, Any]:
        # 解析输入文本
        raw_input = data.get('input_text', '')
        if not raw_input:
            raise ValueError("输入不能为空")
        
        # 先清理输入文本的前后空白
        raw_input = raw_input.strip()
        
        # 如果[NO_SIMU]标签不存在，直接在最前面添加[NO_SIMU]
        tag_pattern = re.compile(r'\[(no[\s_]*(simu))\]', re.IGNORECASE)
        if not tag_pattern.search(raw_input):
            raise ValueError("无法找到有效的非模拟桩信息标签[No Simu]或[no_simu]")
        
        if 'END' not in raw_input and 'end' not in raw_input:
            raw_input += '\nEND;'
            
        # 将多个\n替换为\n
        raw_input = re.sub(r'\n+', '\n', raw_input)
        
        # 解析[NO_SIMU] 到 END; 或end;之间的内容，更灵活地处理空白和缩进
        no_simu_pattern = r'\s*\[(no[\s_]*(simu))\]\s*(.*?)\s*(?:END|end)'
        match = re.search(no_simu_pattern, raw_input, re.DOTALL | re.IGNORECASE)
        
        if match:
            content = match.group(3).strip()
        else:
            raise ValueError("无法找到有效的非模拟桩信息块")
        
        # 按行分割内容
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # 解析KCTR列表
        kctr_values = list(map(int, re.findall(r'-?\d+', lines[0])))
        
        # 解析每种桩类型的信息
        pile_types = {}
        current_line = 1
        
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
            values = list(map(float, re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', pile_data.split('\n')[0]))) # 必须在第一行
            
            # 解析基本字段
            ksh = int(values[0])
            ksu = int(values[1])
            agl = values[2:5]
            
            # 解析地上桩段
            values = list(map(float, re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', '\n'.join(pile_data.split('\n')[1:])))) # 必须在第二行
            nfr = int(values[0])
            idx = 1
            above_ground_sections = []
            for _ in range(nfr):
                if idx + 2 >= len(values):
                    raise ValueError(f"桩类型{pile_type}的地上桩段信息不完整")
                above_ground_sections.append(AboveGroundSection(
                    HFR=values[idx],
                    DOF=values[idx+1],
                    NSF=int(values[idx+2])
                ))
                idx += 3
            
            # 解析地下桩段
            # 如果地上段有nfr个桩段，则地下段从nfr+1开始
            values = list(map(float, re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', '\n'.join(pile_data.split('\n')[1 + nfr + (nfr == 0):])))) # 必须在第1 + nfr + (nfr == 0)行及以后
            nbl = int(values[0])
            idx = 1
            below_ground_sections = []
            for _ in range(nbl):
                if idx + 4 >= len(values):
                    raise ValueError(f"桩类型{pile_type}的地下桩段信息不完整")
                below_ground_sections.append(BelowGroundSection(
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
            pile_types[pile_type] = NoSimuPileModel(
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
        
        # 构建NoSimuModel
        data['no_simu'] = NoSimuModel(
            KCTR=kctr_values,
            pile_types=pile_types
        )
        
        return data

# 工厂函数：根据输入文本创建适当的模型实例
def parse_no_simu_text(input_text: str) -> NoSimuInfoModel:
    return NoSimuInfoModel(input_text=input_text)


if __name__ == "__main__":
    # 测试输入
    input_text = """
    [NO_SIMU]
    1 2 1 3 1
    <1>
    0 2 0.0 0.0 1.0
    1 5.0 1.0 2
    2 10.0 1.0 30000.0 30.0 3
        15.0 1.0 28000.0 28.0 3
    25000.0 30000000.0 1.0
    <2>
    1 3 0.0 0.0 1.0
    0
    1 10.0 1.0 35000.0 30.0 2
    30000.0 32000000.0 1.0
    <3>
    0 1 0.0 0.0 1.0
    1 5.0 1.0 2
    1 8.0 0.8 25000.0 30.0 3
    20000.0 28000000.0 1.0
    END;
    """
    
    input_text = """
[no_simu]
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
<0>
0 1 0 0 1
0 0 0 0
7 21.6 1.8 3000 13.7 22
25.7 1.8 5000 15.5 26
14.1 1.8 5000 30.0 15
1.0 1.8 5000 17.0 1
5.9 1.8 10000 17.0 6
2.5 1.8 5000 17.0 3
14.2 1.8 5000 31.0 15
5000 3e7 1
end
    """

    input_text = """
        [NO_SIMU]
        4
        <4>
        0 2 0.0 0.0 1.0
        0
        1 10.0 1.0 30000.0 30.0 3
        5000 3e7 1
        END;
        """
    
    model = parse_no_simu_text(input_text)
    print(f"KCTR({model.no_simu.model_fields['KCTR'].description}): {model.no_simu.KCTR}")
    print(f"桩类型数量: {len(model.no_simu.pile_types)}")
    for type_id, pile_type in model.no_simu.pile_types.items():
        print(f"桩类型 {type_id}:")
        print(f"  KSH({pile_type.model_fields['KSH'].description}): {pile_type.KSH}, KSU({pile_type.model_fields['KSU'].description}): {pile_type.KSU}")
        print(f"  余弦值: {pile_type.AGL}")
        print(f"  地上桩段数: {pile_type.NFR}")
        print(f"  地下桩段数: {pile_type.NBL}")
        
        if pile_type.NFR > 0:
            print("  地上桩段:")
            for section in pile_type.above_ground_sections:
                print(f"    HFR({section.model_fields['HFR'].description}): {section.HFR}, DOF({section.model_fields['DOF'].description}): {section.DOF}, NSF({section.model_fields['NSF'].description}): {section.NSF}")
        if pile_type.NBL > 0:
            print("  地下桩段:")
            for section in pile_type.below_ground_sections:
                print(f"    HBL({section.model_fields['HBL'].description}): {section.HBL:.2f}", end=', ')
                print(f"DOB({section.model_fields['DOB'].description}): {section.DOB:.2f}", end=', ')
                print(f"PMT({section.model_fields['PMT'].description}): {section.PMT:.2f}", end=', ')
                print(f"PFI({section.model_fields['PFI'].description}): {section.PFI:.2f}", end=', ')
                print(f"NSG({section.model_fields['NSG'].description}): {section.NSG}")
        
        print("桩尖:")
        print(f"  PMB({pile_type.model_fields['PMB'].description}): {pile_type.PMB:.2f}", end=', ')
        print(f"PEH({pile_type.model_fields['PEH'].description}): {pile_type.PEH:.2f}", end=', ')
        print(f"PKE({pile_type.model_fields['PKE'].description}): {pile_type.PKE:.2f}")
