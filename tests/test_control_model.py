#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
控制模型的pytest测试文件
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bcad_pile.models.control_model import create_control_model, ControlModel


@pytest.fixture
def control_model_inputs() -> dict:
    """提供不同测试场景的输入数据"""
    return {
        "jctr1_keywords": """
        [CONTRAL] 
        JCTR = 1
        NACT 2 
        10.5 20.3
        100.0 200.0 300.0 150.0 250.0 350.0
        15.7 25.9
        120.0 220.0 320.0 170.0 270.0 370.0
        """,
        
        "jctr2": """
        [CONTRAL] 
        JCTR = 2
        """,
        
        "jctr3_keywords": """
        [CONTRAL] 
        JCTR = 3
        INO = 5
        """,
        
        "jctr_implicit": """
        [CONTROL] 
        1
        NACT 2 
        10.5 20.3
        100.0 200.0 300.0 150.0 250.0 350.0
        15.7 25.9
        120.0 220.0 320.0 170.0 270.0 370.0
        """,
        
        "sequential_jctr1": """
        [CONTROL] 
        1
        2
        10.5 20.3
        100.0 200.0 300.0 150.0 250.0 350.0
        15.7 25.9
        120.0 220.0 320.0 170.0 270.0 370.0
        """,
        
        "sequential_jctr3": """
        [CONTROL] 
        3
        5
        """,
        
        "invalid_jctr": """
        [CONTRAL] 
        JCTR = 4
        """,
        
        "jctr1_missing_nact": """
        [CONTROL]
        JCTR = 1
        """,
        
        "jctr3_missing_ino": """
        [CONTROL]
        JCTR = 3
        """
    }


class TestControlModel:
    """控制模型的测试类"""
    
    def test_jctr_1_with_keywords(self, control_model_inputs: dict):
        """测试使用关键词的JCTR=1情况"""
        model = create_control_model(control_model_inputs["jctr1_keywords"])
        assert model.control.JCTR == 1
        assert model.control.NACT == 2
        assert len(model.control.force_points) == 2
        
        # 测试第一个作用点
        point1 = model.control.force_points[0]
        assert point1.X == 10.5
        assert point1.Y == 20.3
        assert point1.FX == 100.0
        assert point1.FY == 200.0
        assert point1.FZ == 300.0
        assert point1.MX == 150.0
        assert point1.MY == 250.0
        assert point1.MZ == 350.0
        
        # 测试第二个作用点
        point2 = model.control.force_points[1]
        assert point2.X == 15.7
        assert point2.Y == 25.9
        assert point2.FX == 120.0
        assert point2.FY == 220.0
        assert point2.FZ == 320.0
        assert point2.MX == 170.0
        assert point2.MY == 270.0
        assert point2.MZ == 370.0
    
    def test_jctr_2(self, control_model_inputs: dict):
        """测试JCTR=2情况"""
        model = create_control_model(control_model_inputs["jctr2"])
        assert model.control.JCTR == 2
    
    def test_jctr_3_with_keywords(self, control_model_inputs: dict):
        """测试使用关键词的JCTR=3情况"""
        model = create_control_model(control_model_inputs["jctr3_keywords"])
        assert model.control.JCTR == 3
        assert model.control.INO == 5
    
    def test_jctr_implicit_with_keywords(self, control_model_inputs: dict):
        """测试无显式JCTR字段但使用其他关键词的情况"""
        model = create_control_model(control_model_inputs["jctr_implicit"])
        assert model.control.JCTR == 1
        assert model.control.NACT == 2
        assert len(model.control.force_points) == 2
    
    def test_sequential_parsing_jctr_1(self, control_model_inputs: dict):
        """测试顺序解析JCTR=1情况（不使用NACT关键词）"""
        model = create_control_model(control_model_inputs["sequential_jctr1"])
        assert model.control.JCTR == 1
        assert model.control.NACT == 2
        assert len(model.control.force_points) == 2
        
        # 测试第一个作用点
        point1 = model.control.force_points[0]
        assert point1.X == 10.5
        assert point1.Y == 20.3
        assert point1.FX == 100.0
        assert point1.FY == 200.0
        assert point1.FZ == 300.0
        assert point1.MX == 150.0
        assert point1.MY == 250.0
        assert point1.MZ == 350.0
    
    def test_sequential_parsing_jctr_3(self, control_model_inputs: dict):
        """测试顺序解析JCTR=3情况（不使用INO关键词）"""
        model = create_control_model(control_model_inputs["sequential_jctr3"])
        assert model.control.JCTR == 3
        assert model.control.INO == 5
    
    def test_invalid_jctr(self, control_model_inputs: dict):
        """测试无效的JCTR值"""
        with pytest.raises(ValueError, match="JCTR 必须是 1, 2 或 3"):
            create_control_model(control_model_inputs["invalid_jctr"])
    
    # def test_missing_control_tag(self, control_model_inputs: dict):
    #     """测试缺少[CONTROL]标签的情况"""
    #     with pytest.raises(ValueError, match="无法解析JCTR值，未找到\[CONTROL\]或\[CONTRAL\]标签"):
    #         create_control_model(control_model_inputs["missing_control_tag"])
    
    def test_jctr_1_missing_nact(self, control_model_inputs: dict):
        """测试JCTR=1但缺少NACT的情况"""
        with pytest.raises(ValueError, match="JCTR=1时必须提供NACT值"):
            create_control_model(control_model_inputs["jctr1_missing_nact"])
    
    def test_jctr_3_missing_ino(self, control_model_inputs: dict):
        """测试JCTR=3但缺少INO的情况"""
        with pytest.raises(ValueError, match="JCTR=3时必须提供INO参数"):
            create_control_model(control_model_inputs["jctr3_missing_ino"])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
