#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
非模拟桩信息模型的pytest测试文件
"""

import pytest
import sys
from pathlib import Path
from pydantic import ValidationError

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pypile.models.no_simu_model import parse_no_simu_text, NoSimuInfoModel


@pytest.fixture
def no_simu_model_inputs() -> dict:
    """提供不同测试场景的输入数据"""
    return {
        "standard": """
        [no_simu]
        1 2 1 3 1
        <1>
        0 2 0.0 0.0 1.0
        1 5.0 1.0 2
        2 10.0 1.0 30000.0 30.0 3 15.0 1.0 28000.0 28.0 3
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
        """,
        
        "single_type": """
        [NO_SIMU]
        1 1 1
        <1>
        0 2 0.0 0.0 1.0
        1 5.0 1.0 2
        1 10.0 1.0 30000.0 30.0 3
        25000.0 30000000.0 1.0
        END;
        """,
        
        "no_above_ground": """
        [NO_SIMU]
        4
        <4>
        0 2 0.0 0.0 1.0
        0
        1 10.0 1.0 30000.0 30.0 3
        5000 3e7 1
        END;
        """,
        
        "invalid_ksh": """
        [NO_SIMU]
        5
        <5>
        2 2 0.0 0.0 1.0
        1 5.0 1.0 2
        1 10.0 1.0 30000.0 30.0 3
        5000 3e7 1
        END;
        """,
        
        "invalid_ksu": """
        [NO_SIMU]
        6
        <6>
        0 5 0.0 0.0 1.0
        1 5.0 1.0 2
        1 10.0 1.0 30000.0 30.0 3
        5000 3e7 1
        END;
        """,
        
        "invalid_agl": """
        [NO_SIMU]
        7
        <7>
        0 2 0.0 0.0 0.5
        1 5.0 1.0 2
        1 10.0 1.0 30000.0 30.0 3
        25000.0 30000000.0 1.0
        END;
        """,
        
        "missing_tag": """
        1 1 1 1 1
        <1>
        0 2 0.0 0.0 1.0
        1 5.0 1.0 2
        2 10.0 1.0 30000.0 30.0 5
        15.0 1.0 28000.0 28.0 3
        25000.0 30000000.0 1.0
        END;
        """,
        
        "missing_end": """
        [NO_SIMU]
        1 1 1 1 1
        <1>
        0 2 0.0 0.0 1.0
        1 5.0 1.0 2
        2 10.0 1.0 30000.0 30.0 5
        15.0 1.0 28000.0 28.0 3
        25000.0 30000000.0 1.0
        """,
        
        "invalid_pfi": """
        [NO_SIMU]
        8
        <8>
        0 2 0.0 0.0 1.0
        1 5.0 1.0 2
        1 10.0 1.0 30000.0 95.0 3
        25000.0 30000000.0 1.0
        END;
        """,
        
        "inconsistent_type_id": """
        [NO_SIMU]
        10
        <9>
        0 2 0.0 0.0 1.0
        1 5.0 1.0 2
        1 10.0 1.0 30000.0 30.0 3
        25000.0 30000000.0 1.0
        END;
        """
    }


class TestNoSimuModel:
    """非模拟桩信息模型的测试类"""
    
    def test_standard_no_simu(self, no_simu_model_inputs: dict):
        """测试标准的非模拟桩信息"""
        model = parse_no_simu_text(no_simu_model_inputs["standard"])
        
        # 验证KCTR列表
        assert model.no_simu.KCTR == [1, 2, 1, 3, 1]
        
        # 验证桩类型数量
        assert len(model.no_simu.pile_types) == 3
        
        # 验证第一种桩类型信息
        pile_type_1 = model.no_simu.pile_types[1]
        assert pile_type_1.KSH == 0
        assert pile_type_1.KSU == 2
        assert pile_type_1.AGL == [0.0, 0.0, 1.0]
        assert pile_type_1.NFR == 1
        assert len(pile_type_1.above_ground_sections) == 1
        assert pile_type_1.NBL == 2
        assert len(pile_type_1.below_ground_sections) == 2
        assert pile_type_1.PMB == 25000.0
        assert pile_type_1.PEH == 30000000.0
        assert pile_type_1.PKE == 1.0
        
        # 验证第一个地上桩段
        above_section_1 = pile_type_1.above_ground_sections[0]
        assert above_section_1.HFR == 5.0
        assert above_section_1.DOF == 1.0
        assert above_section_1.NSF == 2
        
        # 验证第一个地下桩段
        below_section_1 = pile_type_1.below_ground_sections[0]
        assert below_section_1.HBL == 10.0
        assert below_section_1.DOB == 1.0
        assert below_section_1.PMT == 30000.0
        assert below_section_1.PFI == 30.0
        assert below_section_1.NSG == 3
    
    def test_single_type(self, no_simu_model_inputs: dict):
        """测试单一桩类型的情况"""
        model = parse_no_simu_text(no_simu_model_inputs["single_type"])
        assert model.no_simu.KCTR == [1, 1, 1]
        assert len(model.no_simu.pile_types) == 1
        
        # 验证桩类型信息
        pile_type = model.no_simu.pile_types[1]
        assert pile_type.NFR == 1
        assert pile_type.NBL == 1
    
    def test_no_above_ground(self, no_simu_model_inputs: dict):
        """测试无地上桩段的情况"""
        model = parse_no_simu_text(no_simu_model_inputs["no_above_ground"])
        assert model.no_simu.KCTR == [4]
        assert len(model.no_simu.pile_types) == 1
        
        # 验证桩类型信息
        pile_type = model.no_simu.pile_types[4]
        assert pile_type.NFR == 0
        assert len(pile_type.above_ground_sections) == 0
        assert pile_type.NBL == 1
        assert len(pile_type.below_ground_sections) == 1
    
    def test_invalid_ksh(self, no_simu_model_inputs: dict):
        """测试无效的KSH值"""
        with pytest.raises(ValueError, match="KSH必须是0\\(圆截面\\)或1\\(方截面\\)"):
            parse_no_simu_text(no_simu_model_inputs["invalid_ksh"])
    
    def test_invalid_ksu(self, no_simu_model_inputs: dict):
        """测试无效的KSU值"""
        with pytest.raises(ValueError, match="KSU必须是1\\(摩擦桩\\), 2\\(摩擦端承桩\\), 3\\(端承桩\\)或4\\(其他边界条件\\)"):
            parse_no_simu_text(no_simu_model_inputs["invalid_ksu"])
    
    def test_invalid_agl(self, no_simu_model_inputs: dict):
        """测试无效的AGL值"""
        with pytest.raises(ValueError, match="方向余弦的平方和必须等于1"):
            parse_no_simu_text(no_simu_model_inputs["invalid_agl"])
    
    def test_missing_tag(self, no_simu_model_inputs: dict):
        """测试缺少[NO_SIMU]标签的情况"""
        # 缺少标签应该会报错
        with pytest.raises(ValidationError) as excinfo:
            parse_no_simu_text(no_simu_model_inputs["missing_tag"])
        assert "无法找到有效的非模拟桩信息标签[No Simu]或[no_simu]" in str(excinfo.value)
    
    def test_missing_end(self, no_simu_model_inputs: dict):
        """测试缺少END;标签的情况"""
        # 缺少END;应该被自动添加
        model = parse_no_simu_text(no_simu_model_inputs["missing_end"])
        assert model.no_simu.KCTR == [1, 1, 1, 1, 1]
    
    def test_invalid_pfi(self, no_simu_model_inputs: dict):
        """测试无效的土内摩擦角值"""
        with pytest.raises(ValueError, match="土内摩擦角必须在0-90度范围内"):
            parse_no_simu_text(no_simu_model_inputs["invalid_pfi"])
    
    def test_inconsistent_type_id(self, no_simu_model_inputs: dict):
        """测试桩类型ID与KCTR不匹配的情况"""
        with pytest.raises(ValidationError) as excinfo:
            parse_no_simu_text(no_simu_model_inputs["inconsistent_type_id"])
        assert "KCTR(非模拟桩类型标识)中存在值:10,但并未找到对应的土层信息，请检查!" in str(excinfo.value)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
