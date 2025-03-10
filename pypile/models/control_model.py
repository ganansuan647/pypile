from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Union, Tuple, Dict, Any, ClassVar
import re

# 基础控制模型
class ControlModel(BaseModel):
    JCTR: int = Field(..., description="控制参数，定义执行的分析类型")
    
    @field_validator('JCTR')
    @classmethod
    def validate_jctr(cls, v: int) -> int:
        if v not in [1, 2, 3]:
            raise ValueError("JCTR 必须是 1, 2 或 3")
        return v

# JCTR=3 的情况，需要额外的 INO 参数
class Control3Model(ControlModel):
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
class Control1Model(ControlModel):
    NACT: int = Field(..., description="作用力点数量")
    force_points: List[ForcePoint] = Field(..., description="作用力点列表")

# 主控制模型，根据JCTR值动态选择适当的模型
class ControlInfoModel(BaseModel):
    control: Union[ControlModel, Control1Model, Control3Model]
    
    @model_validator(mode='before')
    @classmethod
    def parse_input(cls, data: Any) -> Dict[str, Any]:
        # 解析输入文本
        raw_input = data.get('input_text', '')
        if not raw_input:
            raise ValueError("输入不能为空")
        
        # 如果[CONTRAL] 或 [CONTROL]标签不存在,报错
        if not re.search(r'\[\s*(?:CONTROL|CONTRAL)\s*\]', raw_input, re.IGNORECASE | re.DOTALL):
            raise ValueError("无法找到有效的控制信息标签[Control]")
        
        # 使用正则表达式检查任意大小写的END
        if not re.search(r'END;', raw_input, re.IGNORECASE):
            raw_input += '\nEND;'
            
        # 将多个\n替换为\n
        raw_input = re.sub(r'\n+', '\n', raw_input)
        
        # 解析[CONTROL]或[CONTRAL] 到 END; 之间的内容作为input_text
        control_pattern = r'\[\s*(?:CONTROL|CONTRAL)\s*\].*?(?:END;)'
        match = re.search(control_pattern, raw_input, re.IGNORECASE | re.DOTALL)

        if match:
            input_text = match.group(0)
        else:
            raise ValueError("无法找到有效的控制块")
        
        # 预处理：提取所有数值，用于后续处理
        all_numbers = re.findall(r'-?\d+\.?\d*', input_text)
        all_numbers = [n for n in all_numbers if n.strip()]  # 过滤空字符串
        
        # 尝试多种方式提取JCTR值
        jctr = None
        
        # 方法1：通过JCTR=n格式提取
        jctr_match = re.search(r'JCTR\s*=\s*(\d+)', input_text, re.IGNORECASE)
        if jctr_match:
            jctr = int(jctr_match.group(1))
        
        # 方法2：通过[CONTROL]或[CONTRAL]后第一个数字提取
        if jctr is None:
            control_match = re.search(r'\[\s*(?:CONTROL|CONTRAL)\s*\]', input_text, re.IGNORECASE)
            if control_match:
                post_control_text = input_text[control_match.end():]
                first_int_match = re.search(r'\s*(\d+)', post_control_text)
                if first_int_match:
                    jctr = int(first_int_match.group(1))
        
        # 方法3：如果没有控制标签，尝试使用第一个数字作为JCTR
        if jctr is None and all_numbers:
            try:
                jctr = int(float(all_numbers[0]))
            except (ValueError, IndexError):
                pass
        
        # 如果仍然找不到JCTR，那么报错
        if jctr is None:
            raise ValueError("无法解析JCTR值，请检查输入格式")
        
        # 验证JCTR值
        if jctr not in [1, 2, 3]:
            raise ValueError("JCTR 必须是 1, 2 或 3")
            
        # 获取JCTR之后的文本内容
        post_control_text = ""
        
        # 通过提取JCTR值后的文本
        if jctr_match:
            # 如果使用JCTR=格式，获取匹配后的文本
            post_control_text = input_text[jctr_match.end():].strip()
        elif control_match and first_int_match:
            # 如果是通过控制标签后第一个数字，获取该数字后的文本
            post_control_start = control_match.end() + len(first_int_match.group(0))
            post_control_text = input_text[post_control_start:].strip()
        else:
            # 如果是提取第一个数字，获取该数字后的文本
            # 找到第一个数字在原文本中的位置
            if all_numbers:
                number_pos = input_text.find(all_numbers[0])
                if number_pos >= 0:
                    number_end_pos = number_pos + len(all_numbers[0])
                    post_control_text = input_text[number_end_pos:].strip()
        
        # JCTR=1 的处理逻辑
        if jctr == 1:
            # 尝试从post_control_text中提取NACT值
            nact = None
            
            # 方法1：通过NACT关键词提取
            nact_match = re.search(r'NACT\s+(\d+)', post_control_text, re.IGNORECASE)
            if nact_match:
                nact = int(nact_match.group(1))
                post_nact_text = post_control_text[nact_match.end():].strip()
            else:
                # 方法2：尝试提取JCTR后的第一个数字作为NACT
                remaining_numbers = re.findall(r'-?\d+\.?\d*', post_control_text)
                if remaining_numbers:
                    try:
                        nact = int(float(remaining_numbers[0]))
                        # 更新post_control_text，移除NACT值
                        nact_pos = post_control_text.find(remaining_numbers[0])
                        if nact_pos >= 0:
                            post_nact_text = post_control_text[nact_pos + len(remaining_numbers[0]):].strip()
                        else:
                            post_nact_text = post_control_text
                    except (ValueError, IndexError):
                        nact = None
                        post_nact_text = post_control_text
                else:
                    nact = None
                    post_nact_text = post_control_text
            
            if nact is None:
                raise ValueError("JCTR=1时必须提供NACT值")
                
            # 提取剩余数值作为力学作用点信息
            forces_values = re.findall(r'-?\d+\.?\d*', post_nact_text)
            forces_values = [float(v) for v in forces_values if v.strip()]
            
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
            data['control'] = ControlModel(JCTR=jctr)
            
        elif jctr == 3:
            # JCTR=3的情况，需要额外的INO参数
            ino = None
            
            # 方法1：通过INO=格式提取
            ino_match = re.search(r'INO\s*=\s*(\d+)', post_control_text, re.IGNORECASE)
            if ino_match:
                ino = int(ino_match.group(1))
            else:
                # 方法2：尝试提取post_control_text中的第一个数字作为INO
                remaining_numbers = re.findall(r'-?\d+\.?\d*', post_control_text)
                if remaining_numbers:
                    try:
                        ino = int(float(remaining_numbers[0]))
                    except (ValueError, IndexError):
                        pass
            
            if ino is None:
                raise ValueError("JCTR=3时必须提供INO参数")
            
            data['control'] = Control3Model(JCTR=jctr, INO=ino)
        
        return data

# 工厂函数：根据输入文本创建适当的模型实例
def parse_control_text(input_text: str) -> ControlInfoModel:
    return ControlInfoModel(input_text=input_text)


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
        model1 = parse_control_text(input_text1)
        print("JCTR=1 模型验证通过:", model1.control.JCTR == 1)
        print(f"作用点数量: {model1.control.NACT}")
        for i, point in enumerate(model1.control.force_points):
            print(f"作用点 {i+1}: {point}")
        
        model2 = parse_control_text(input_text2)
        print("JCTR=2 模型验证通过:", model2.control.JCTR == 2)
        
        model3 = parse_control_text(input_text3)
        print("JCTR=3 模型验证通过:", model3.control.JCTR == 3)
        print(f"指定桩号: {model3.control.INO}")
        
        model4 = parse_control_text(input_text4)
        print("无显式JCTR字段情况 模型验证通过:", model4.control.JCTR == 1)
        if hasattr(model4.control, 'NACT'):
            print(f"作用点数量: {model4.control.NACT}")
            for i, point in enumerate(model4.control.force_points):
                print(f"作用点 {i+1}: {point}")
        
        model5 = parse_control_text(input_text5)
        print("顺序解析NACT情况 模型验证通过:", model5.control.JCTR == 1)
        if hasattr(model5.control, 'NACT'):
            print(f"作用点数量: {model5.control.NACT}")
            for i, point in enumerate(model5.control.force_points):
                print(f"作用点 {i+1}: {point}")
        
        model6 = parse_control_text(input_text6)
        print("顺序解析INO情况 模型验证通过:", model6.control.JCTR == 3)
        if hasattr(model6.control, 'INO'):
            print(f"指定桩号: {model6.control.INO}")
        
    except Exception as e:
        print(f"验证失败: {e}")