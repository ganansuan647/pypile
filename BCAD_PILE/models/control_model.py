from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Union, Tuple, Dict, Any, ClassVar
import re

# 基础控制模型
class ControlBaseModel(BaseModel):
    JCTR: int = Field(..., description="控制参数，定义执行的分析类型")
    
    @field_validator('JCTR')
    @classmethod
    def validate_jctr(cls, v: int) -> int:
        if v not in [1, 2, 3]:
            raise ValueError("JCTR 必须是 1, 2 或 3")
        return v

# JCTR=3 的情况，需要额外的 INO 参数
class Control3Model(ControlBaseModel):
    INO: int = Field(..., description="指定单桩的桩号")

# 外力作用点模型
class ForcePoint(BaseModel):
    X: float = Field(..., description="作用点X坐标")
    Y: float = Field(..., description="作用点Y坐标") 
    FX: float = Field(..., description="X方向力")
    FY: float = Field(..., description="Y方向力")
    FZ: float = Field(..., description="Z方向力")
    MX: float = Field(..., description="X方向力矩")
    MY: float = Field(..., description="Y方向力矩")
    MZ: float = Field(..., description="Z方向力矩")

# JCTR=1 的情况，需要外力信息
class Control1Model(ControlBaseModel):
    NACT: int = Field(..., description="作用力点数量")
    force_points: List[ForcePoint] = Field(..., description="作用力点列表")

# 主控制模型，根据JCTR值动态选择适当的模型
class ControlModel(BaseModel):
    control: Union[ControlBaseModel, Control1Model, Control3Model]
    
    @model_validator(mode='before')
    @classmethod
    def parse_input(cls, data: Any) -> Dict[str, Any]:
        # 解析输入文本
        raw_input = data.get('input_text', '')
        if not raw_input:
            raise ValueError("输入不能为空")
        
        # 如果[CONTRAL] 或 [CONTROL]标签不存在,直接在最前面添加[CONTROL]
        if '[CONTRAL]' not in raw_input and '[CONTROL]' not in raw_input:
            raw_input = '[CONTROL]\n' + raw_input
        
        # 提取JCTR值
        match = re.search(r'JCTR\s*=\s*(\d+)', raw_input) or re.search(r'\[CONTRAL\]\s+(\d+)', raw_input) or re.search(r'\[CONTROL\]\s+(\d+)', raw_input)
        if not match:
            # 尝试查找[CONTROL]或[CONTRAL]标签，然后查找其后的第一个整数
            control_match = re.search(r'\[(CONTROL|CONTRAL)\]', raw_input)
            if control_match:
                # 找到标签后面的文本
                post_control_text = raw_input[control_match.end():]
                # 查找第一个整数
                first_int_match = re.search(r'\s*(\d+)', post_control_text)
                if first_int_match:
                    jctr = int(first_int_match.group(1))
                    # 更新post_control_text，去掉已解析的JCTR值
                    post_control_text = post_control_text[first_int_match.end():]
                else:
                    raise ValueError("无法在[CONTROL]标签后找到JCTR值")
            else:
                raise ValueError("无法解析JCTR值，未找到[CONTROL]或[CONTRAL]标签")
        else:
            jctr = int(match.group(1))
            # 获取匹配之后的文本
            if '[CONTRAL]' in raw_input or '[CONTROL]' in raw_input:
                control_tag = '[CONTRAL]' if '[CONTRAL]' in raw_input else '[CONTROL]'
                control_pos = raw_input.find(control_tag) + len(control_tag)
                post_match = re.search(r'JCTR\s*=\s*\d+', raw_input[control_pos:])
                if post_match:
                    post_control_text = raw_input[control_pos + post_match.end():]
                else:
                    # 找不到JCTR标记，则尝试找到JCTR值后的文本
                    post_match = re.search(r'\s+\d+', raw_input[control_pos:])
                    if post_match and int(post_match.group().strip()) == jctr:
                        post_control_text = raw_input[control_pos + post_match.end():]
                    else:
                        post_control_text = raw_input[control_pos:]
            else:
                post_jctr = re.search(r'JCTR\s*=\s*\d+', raw_input)
                if post_jctr:
                    post_control_text = raw_input[post_jctr.end():]
                else:
                    post_control_text = raw_input
        
        if jctr == 1:
            # 解析JCTR=1的情况
            nact_match = re.search(r'NACT\s+(\d+)', post_control_text)
            if nact_match:
                # 传统方式：使用关键字NACT
                nact = int(nact_match.group(1))
                forces_text = post_control_text[nact_match.end():].strip()
            else:
                # 顺序解析：查找下一个整数作为NACT
                nact_match = re.search(r'\s*(\d+)', post_control_text)
                if nact_match:
                    nact = int(nact_match.group(1))
                    forces_text = post_control_text[nact_match.end():].strip()
                else:
                    raise ValueError("JCTR=1时必须提供NACT值")
            
            forces_values = list(map(float, re.findall(r'-?\d+\.?\d*', forces_text)))
            
            if len(forces_values) < nact * 8:
                raise ValueError(f"外力信息不完整，需要{nact * 8}个值，但只找到{len(forces_values)}个")
            
            force_points = []
            for i in range(nact):
                idx = i * 8
                force_points.append(ForcePoint(
                    X=forces_values[idx],
                    Y=forces_values[idx+1],
                    FX=forces_values[idx+2],
                    FY=forces_values[idx+3],
                    FZ=forces_values[idx+4],
                    MX=forces_values[idx+5],
                    MY=forces_values[idx+6],
                    MZ=forces_values[idx+7]
                ))
            
            data['control'] = Control1Model(JCTR=jctr, NACT=nact, force_points=force_points)
            
        elif jctr == 2:
            # JCTR=2的情况，只需要JCTR值
            data['control'] = ControlBaseModel(JCTR=jctr)
            
        elif jctr == 3:
            # JCTR=3的情况，需要额外的INO参数
            ino_match = re.search(r'INO\s*=\s*(\d+)', post_control_text) or re.search(r'\[CONTRAL\]\s+3\s+(\d+)', post_control_text)
            if ino_match:
                # 传统方式：使用关键字INO
                ino = int(ino_match.group(1))
            else:
                # 顺序解析：查找下一个整数作为INO
                ino_match = re.search(r'\s*(\d+)', post_control_text)
                if ino_match:
                    ino = int(ino_match.group(1))
                else:
                    raise ValueError("JCTR=3时必须提供INO参数")
            
            data['control'] = Control3Model(JCTR=jctr, INO=ino)
            
        else:
            raise ValueError("JCTR 必须是 1, 2 或 3")
        
        return data

# 工厂函数：根据输入文本创建适当的模型实例
def create_control_model(input_text: str) -> ControlModel:
    return ControlModel(input_text=input_text)


if __name__ == "__main__":
    # JCTR=1 的情况
    input_text1 = """
    [CONTRAL] 
    JCTR = 1
    NACT 2 
    10.5 20.3
    100.0 200.0 300.0 150.0 250.0 350.0
    15.7 25.9
    120.0 220.0 320.0 170.0 270.0 370.0
    """

    # JCTR=2 的情况
    input_text2 = """
    [CONTRAL] 
    JCTR = 2
    """

    # JCTR=3 的情况
    input_text3 = """
    [CONTRAL] 
    JCTR = 3
    INO = 5
    """
    
    # 无显式JCTR字段的情况，直接用第一个数字
    input_text4 = """
    [CONTROL] 
    1
    NACT 2 
    10.5 20.3
    100.0 200.0 300.0 150.0 250.0 350.0
    15.7 25.9
    120.0 220.0 320.0 170.0 270.0 370.0
    """
    
    # 无关键词的顺序解析情况，JCTR=1，不使用NACT关键字
    input_text5 = """
    [CONTROL] 
    1
    2
    10.5 20.3
    100.0 200.0 300.0 150.0 250.0 350.0
    15.7 25.9
    120.0 220.0 320.0 170.0 270.0 370.0
    """
    
    # 无关键词的顺序解析情况，JCTR=3，不使用INO关键字
    input_text6 = """
    [CONTROL] 
    3
    5
    """

    try:
        model1 = create_control_model(input_text1)
        print("JCTR=1 模型验证通过:", model1.control)
        print(f"作用点数量: {model1.control.NACT}")
        print(f"第一个作用点: {model1.control.force_points[0]}")
        
        model2 = create_control_model(input_text2)
        print("JCTR=2 模型验证通过:", model2.control)
        
        model3 = create_control_model(input_text3)
        print("JCTR=3 模型验证通过:", model3.control)
        print(f"指定桩号: {model3.control.INO}")
        
        model4 = create_control_model(input_text4)
        print("无显式JCTR字段情况 模型验证通过:", model4.control)
        if hasattr(model4.control, 'NACT'):
            print(f"作用点数量: {model4.control.NACT}")
            print(f"第一个作用点: {model4.control.force_points[0]}")
        
        model5 = create_control_model(input_text5)
        print("顺序解析NACT情况 模型验证通过:", model5.control)
        if hasattr(model5.control, 'NACT'):
            print(f"作用点数量: {model5.control.NACT}")
            print(f"第一个作用点: {model5.control.force_points[0]}")
        
        model6 = create_control_model(input_text6)
        print("顺序解析INO情况 模型验证通过:", model6.control)
        if hasattr(model6.control, 'INO'):
            print(f"指定桩号: {model6.control.INO}")
        
    except Exception as e:
        print(f"验证失败: {e}")