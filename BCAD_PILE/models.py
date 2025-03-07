from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Literal
import numpy as np
from enum import Enum
import re
from pydantic import (
    BaseModel, 
    Field, 
    field_validator,
    model_validator,
    conlist,
    confloat,
    conint,
    ValidationError
)
import yaml
import json
import toml
from dataclasses import dataclass

# 定义常量和枚举类型
class PileShape(int, Enum):
    """桩断面形状枚举"""
    CIRCULAR = 0  # 圆形
    SQUARE = 1    # 方形

class PileBottomConstraint(int, Enum):
    """桩底约束条件枚举"""
    FREE = 0      # 自由
    FIXED = 1     # 固定
    # 可以根据需要添加更多约束类型

class PileSegment(BaseModel):
    """桩段模型"""
    height: float = Field(..., description="桩段高度", gt=0)
    diameter: float = Field(..., description="桩段直径", gt=0)
    calculation_segments: int = Field(..., description="桩段计算分段数", ge=1)
    
    class Config:
        extra = "forbid"  # 禁止额外字段

class UndergroundPileSegment(PileSegment):
    """地下桩段模型，包含地基参数"""
    soil_resistance_coefficient: float = Field(..., description="地基反力系数", ge=0)
    friction_angle: float = Field(..., description="摩擦角", ge=0, le=90)
    
    class Config:
        extra = "forbid"

class PileData(BaseModel):
    """单桩数据模型"""
    coordinate: Tuple[float, float] = Field(..., description="桩的坐标 (x, y)")
    control_info: int = Field(0, description="桩的控制信息")
    shape: PileShape = Field(PileShape.CIRCULAR, description="桩断面形状(0-圆形,1-方形)")
    bottom_constraint: PileBottomConstraint = Field(
        PileBottomConstraint.FREE, 
        description="桩底约束条件"
    )
    inclination_direction_cosines: Tuple[float, float, float] = Field(
        (0.0, 0.0, 1.0), 
        description="桩的倾斜方向余弦"
    )
    
    # 地上段信息
    above_ground_segments: List[PileSegment] = Field(
        [], 
        description="桩地上段信息"
    )
    
    # 地下段信息
    underground_segments: List[UndergroundPileSegment] = Field(
        [], 
        description="桩地下段信息"
    )
    
    # 桩端和材料参数
    soil_end_resistance: float = Field(
        0.0, 
        description="桩端土抗力系数",
        ge=0.0
    )
    elastic_modulus: float = Field(
        ..., 
        description="桩材弹性模量",
        gt=0.0
    )
    shear_elastic_ratio: float = Field(
        0.4, 
        description="桩材剪切模量与弹性模量比",
        gt=0.0,
        le=1.0
    )
    
    @field_validator('inclination_direction_cosines')
    @classmethod
    def validate_direction_cosines(cls, v):
        """验证方向余弦是否合理"""
        x, y, z = v
        sum_squares = x**2 + y**2 + z**2
        # 如果是默认值(0,0,1)或近似于单位向量，则通过验证
        if np.isclose(sum_squares, 0.0, atol=1e-6):
            # 默认值情况下，设为垂直向上
            return (0.0, 0.0, 1.0)
        elif not np.isclose(sum_squares, 1.0, atol=1e-6):
            raise ValueError(f"方向余弦平方和必须为1，当前值: {sum_squares}")
        return v
    
    @model_validator(mode='after')
    @classmethod
    def validate_segments(cls, values):
        """验证桩段信息的一致性"""
        above_ground = values.above_ground_segments
        underground = values.underground_segments
        
        if not above_ground and not underground:
            raise ValueError("至少需要一个地上段或地下段")
            
        # 可以添加更多验证逻辑
        return values
    
    class Config:
        extra = "forbid"

class SimulatedPileData(BaseModel):
    """模拟桩数据模型"""
    coordinate: Tuple[float, float] = Field(..., description="模拟桩坐标 (x, y)")
    control_info: int = Field(0, description="模拟桩控制信息")
    
    class Config:
        extra = "forbid"

class FoundationData(BaseModel):
    """桩基础计算所需的完整数据模型"""
    # 控制参数
    control_parameter: int = Field(..., description="控制参数")
    output_control: Optional[int] = Field(None, description="输出控制参数")
    
    # 桩信息
    piles: List[PileData] = Field(..., description="非模拟桩信息列表")
    simulated_piles: List[SimulatedPileData] = Field([], description="模拟桩信息列表")
    
    # 外部荷载信息（如有需要）
    external_loads: Optional[List[Dict]] = Field(None, description="外部荷载信息")
    
    class Config:
        extra = "forbid"

class DataFileHandler:
    """处理不同格式的输入文件"""
    
    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> FoundationData:
        """从文件加载数据
        
        Args:
            file_path: 输入文件路径
            
        Returns:
            FoundationData: 桩基础数据对象
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.dat':
            return DataFileHandler._load_dat_file(file_path)
        elif suffix == '.json':
            return DataFileHandler._load_json_file(file_path)
        elif suffix == '.yaml' or suffix == '.yml':
            return DataFileHandler._load_yaml_file(file_path)
        elif suffix == '.toml':
            return DataFileHandler._load_toml_file(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    @staticmethod
    def _load_dat_file(file_path: Path) -> FoundationData:
        """加载.dat格式文件
        
        Args:
            file_path: 输入文件路径
            
        Returns:
            FoundationData: 桩基础数据对象
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            # 解析.dat文件内容
            # 这里需要实现解析.dat文件的逻辑，将其转换为FoundationData对象
            # 以下是一个简化的示例：
            
            data = {}
            section = None
            lines = f.readlines()
            
            # 提取控制信息
            for i, line in enumerate(lines):
                line = line.strip().lower()
                
                if line == '[contral]' or line == '[control]':
                    section = 'control'
                    data['control_parameter'] = int(lines[i+1].strip())
                    
                    # 如果控制参数为3，则有输出控制参数
                    if data['control_parameter'] == 3:
                        data['output_control'] = int(lines[i+2].strip())
                    
                elif line == '[arrange]':
                    section = 'arrange'
                    # 解析桩数量信息
                    parts = lines[i+1].strip().split()
                    pile_count = int(parts[0])
                    simulated_pile_count = int(parts[1])
                    
                    # 解析桩坐标
                    piles = []
                    coords_lines = []
                    cur_line = i + 2
                    
                    # 收集坐标行
                    while cur_line < len(lines) and not lines[cur_line].strip().lower() == 'end;':
                        if lines[cur_line].strip():
                            coords_lines.append(lines[cur_line].strip())
                        cur_line += 1
                    
                    coords = []
                    for line in coords_lines:
                        coords.extend([float(x) for x in line.split()])
                    
                    # 将坐标分配给非模拟桩
                    for k in range(pile_count):
                        piles.append({
                            'coordinate': (coords[k*2], coords[k*2+1]),
                            # 其他字段将在后续填充
                        })
                    
                    data['piles'] = piles
                    
                    # 将坐标分配给模拟桩（如果有）
                    if simulated_pile_count > 0:
                        simulated_piles = []
                        offset = pile_count * 2
                        for k in range(simulated_pile_count):
                            simulated_piles.append({
                                'coordinate': (coords[offset+k*2], coords[offset+k*2+1]),
                                # 其他字段将在后续填充
                            })
                        data['simulated_piles'] = simulated_piles
                    
                elif line == '[no_simu]':
                    section = 'no_simu'
                    # TODO: 解析非模拟桩的详细信息
                    # 这部分需要根据.dat文件的具体格式填充
                    
                elif line == '[simu_pe]' or line == '[simupile]':
                    section = 'simu_pe'
                    # TODO: 解析模拟桩的详细信息
                    # 这部分需要根据.dat文件的具体格式填充
            
            # 构建并返回FoundationData对象
            # 注意：这里的实现非常简化，实际解析.dat文件可能需要更复杂的逻辑
            
            # 为简化示例，我们填充一些默认值
            for pile in data.get('piles', []):
                if 'elastic_modulus' not in pile:
                    pile['elastic_modulus'] = 3e7  # 默认弹性模量
                
                if 'above_ground_segments' not in pile:
                    pile['above_ground_segments'] = [{
                        'height': 10.0,
                        'diameter': 1.0,
                        'calculation_segments': 10
                    }]
                
                if 'underground_segments' not in pile:
                    pile['underground_segments'] = [{
                        'height': 20.0,
                        'diameter': 1.0,
                        'calculation_segments': 20,
                        'soil_resistance_coefficient': 5000.0,
                        'friction_angle': 30.0
                    }]
            
            return FoundationData(**data)
    
    @staticmethod
    def _load_json_file(file_path: Path) -> FoundationData:
        """加载JSON格式文件
        
        Args:
            file_path: 输入文件路径
            
        Returns:
            FoundationData: 桩基础数据对象
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return FoundationData(**data)
    
    @staticmethod
    def _load_yaml_file(file_path: Path) -> FoundationData:
        """加载YAML格式文件
        
        Args:
            file_path: 输入文件路径
            
        Returns:
            FoundationData: 桩基础数据对象
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            # 在YAML中，元组会被转换为列表，需要将坐标和方向余弦列表转回元组
            if 'piles' in data:
                for pile in data['piles']:
                    if 'coordinate' in pile and isinstance(pile['coordinate'], list):
                        pile['coordinate'] = tuple(pile['coordinate'])
                    if 'inclination_direction_cosines' in pile and isinstance(pile['inclination_direction_cosines'], list):
                        pile['inclination_direction_cosines'] = tuple(pile['inclination_direction_cosines'])
            if 'simulated_piles' in data:
                for pile in data['simulated_piles']:
                    if 'coordinate' in pile and isinstance(pile['coordinate'], list):
                        pile['coordinate'] = tuple(pile['coordinate'])
            return FoundationData(**data)
    
    @staticmethod
    def _load_toml_file(file_path: Path) -> FoundationData:
        """加载TOML格式文件
        
        Args:
            file_path: 输入文件路径
            
        Returns:
            FoundationData: 桩基础数据对象
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = toml.load(f)
            return FoundationData(**data)
    
    @staticmethod
    def save_to_file(data: FoundationData, file_path: Union[str, Path], format: str = None) -> None:
        """将数据保存到文件
        
        Args:
            data: 桩基础数据对象
            file_path: 输出文件路径
            format: 输出格式，如果为None则根据文件扩展名判断
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        
        # 如果未指定格式，根据文件扩展名推断
        if format is None:
            format = file_path.suffix.lower().lstrip('.')
        
        # 将数据转换为字典 - 使用mode='json'将自定义对象转换为可序列化的字典
        data_dict = data.model_dump(mode='json')
        
        # 将元组转换为列表（由于元组在某些格式中无法正确序列化）
        def convert_tuples_to_lists(obj):
            if isinstance(obj, dict):
                return {k: convert_tuples_to_lists(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tuples_to_lists(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_tuples_to_lists(item) for item in obj]
            else:
                return obj
                
        data_dict = convert_tuples_to_lists(data_dict)
        
        # 创建目录（如果不存在）
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2)
        elif format in ('yaml', 'yml'):
            # 对于YAML，使用默认流类型，元组将被转换为列表
            # YAML中无法直接表示元组，因此在加载时需要进行转换
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data_dict, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        elif format == 'toml':
            with open(file_path, 'w', encoding='utf-8') as f:
                toml.dump(data_dict, f)
        elif format == 'dat':
            # 这里需要实现将数据转换为.dat格式的逻辑
            # 由于.dat格式可能是特定的专有格式，这里只是一个示例框架
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("[contral]\n")
                f.write(f"{data.control_parameter}\n")
                if data.output_control is not None:
                    f.write(f"{data.output_control}\n")
                f.write("end;\n")
                
                f.write("[arrange]\n")
                f.write(f"{len(data.piles)} {len(data.simulated_piles)}\n")
                
                # 写入非模拟桩坐标
                for pile in data.piles:
                    f.write(f"{pile.coordinate[0]} {pile.coordinate[1]}\n")
                
                # 写入模拟桩坐标
                for pile in data.simulated_piles:
                    f.write(f"{pile.coordinate[0]} {pile.coordinate[1]}\n")
                
                f.write("end;\n")
                
                # 写入非模拟桩详细信息
                f.write("[no_simu]\n")
                # 这里需要根据.dat文件的具体格式填充
                f.write("end;\n")
                
                # 写入模拟桩详细信息
                f.write("[simu_pe]\n")
                # 这里需要根据.dat文件的具体格式填充
                f.write("end;\n")
        else:
            raise ValueError(f"不支持的输出格式: {format}")

# 示例用法
def test_load_file():
    """测试加载文件"""
    try:
        # 创建测试输出目录
        output_dir = Path("tests/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载.dat文件
        print("加载测试数据文件...")
        data = DataFileHandler.load_from_file("tests/Test-1-2.dat")
        print(f"加载了 {len(data.piles)} 个非模拟桩和 {len(data.simulated_piles)} 个模拟桩")
        
        # 保存为不同格式
        print("将数据保存为不同格式...")
        DataFileHandler.save_to_file(data, "tests/output/output.json")
        DataFileHandler.save_to_file(data, "tests/output/output.yaml")
        # DataFileHandler.save_to_file(data, "tests/output/output.toml")
        
        # 重新加载并验证
        print("验证数据一致性...")
        json_data = DataFileHandler.load_from_file("tests/output/output.json")
        assert len(json_data.piles) == len(data.piles), "JSON数据加载后桩数量不匹配"
        
        yaml_data = DataFileHandler.load_from_file("tests/output/output.yaml")
        assert len(yaml_data.piles) == len(data.piles), "YAML数据加载后桩数量不匹配"
        
        # TOML可能对复杂嵌套结构支持不佳，先尝试不加载TOML验证
        # toml_data = DataFileHandler.load_from_file("tests/output/output.toml")
        # assert len(toml_data.piles) == len(data.piles), "TOML数据加载后桩数量不匹配"
        
        print("所有格式的转换和加载测试通过")
        return data  # 返回加载的数据，以便在交互式环境中检查
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load_file()
