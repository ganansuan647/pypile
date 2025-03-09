#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
桩布置信息模型的pytest测试文件
"""

import pytest
import sys
from pathlib import Path
from pydantic import ValidationError

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pypile.models.arrange_model import parse_arrange_text, ArrangeInfoModel


@pytest.fixture
def arrange_model_inputs() -> dict:
    """提供不同测试场景的输入数据"""
    return {
        "standard": """
        [ARRANGE]
        2 1
        0.0 0.0
        5.0 0.0
        10.0 0.0
        END;
        """,
        
        "no_simu_pile": """
        [ARRANGE]
        3 0
        0.0 0.0
        5.0 0.0
        10.0 0.0
        END;
        """,
        
        "no_regular_pile": """
        [ARRANGE]
        0 3
        10.0 0.0
        15.0 0.0
        20.0 0.0
        END;
        """,
        
        "missing_tag": """
        2 1
        0.0 0.0
        5.0 0.0
        10.0 0.0
        END;
        """,
        
        "missing_end": """
        [ARRANGE]
        2 1
        0.0 0.0
        5.0 0.0
        10.0 0.0
        """,
        
        "missing_coordinates": """
        [ARRANGE]
        2 1
        0.0 0.0
        5.0 0.0
        END;
        """,
        
        "missing_values": """
        [ARRANGE]
        END;
        """
    }


class TestArrangeModel:
    """桩布置信息模型的测试类"""
    
    def test_standard_arrange(self, arrange_model_inputs: dict):
        """测试标准的桩布置信息"""
        model = parse_arrange_text(arrange_model_inputs["standard"])
        
        # 验证基本字段
        assert model.arrange.PNUM == 2
        assert model.arrange.SNUM == 1
        
        # 验证非模拟桩坐标
        assert len(model.arrange.pile_coordinates) == 2
        assert model.arrange.pile_coordinates[0].X == 0.0
        assert model.arrange.pile_coordinates[0].Y == 0.0
        assert model.arrange.pile_coordinates[1].X == 5.0
        assert model.arrange.pile_coordinates[1].Y == 0.0
        
        # 验证模拟桩坐标
        assert len(model.arrange.simu_pile_coordinates) == 1
        assert model.arrange.simu_pile_coordinates[0].X == 10.0
        assert model.arrange.simu_pile_coordinates[0].Y == 0.0
    
    def test_no_simu_pile(self, arrange_model_inputs: dict):
        """测试无模拟桩的情况"""
        model = parse_arrange_text(arrange_model_inputs["no_simu_pile"])
        
        assert model.arrange.PNUM == 3
        assert model.arrange.SNUM == 0
        assert len(model.arrange.pile_coordinates) == 3
        assert len(model.arrange.simu_pile_coordinates) == 0
    
    def test_no_regular_pile(self, arrange_model_inputs: dict):
        """测试无非模拟桩的情况"""
        model = parse_arrange_text(arrange_model_inputs["no_regular_pile"])
        
        assert model.arrange.PNUM == 0
        assert model.arrange.SNUM == 3
        assert len(model.arrange.pile_coordinates) == 0
        assert len(model.arrange.simu_pile_coordinates) == 3
    
    def test_missing_tag(self, arrange_model_inputs: dict):
        """测试缺少[ARRANGE]标签的情况"""
        # 缺少标签应该报错
        with pytest.raises(ValidationError) as excinfo:
            parse_arrange_text(arrange_model_inputs["missing_tag"])
        assert "无法找到有效的布置块[ARRANGE]" in str(excinfo.value)
    
    def test_missing_end(self, arrange_model_inputs: dict):
        """测试缺少END;标签的情况"""
        # 缺少END;应该被自动添加
        model = parse_arrange_text(arrange_model_inputs["missing_end"])
        
        assert model.arrange.PNUM == 2
        assert model.arrange.SNUM == 1
        assert len(model.arrange.pile_coordinates) == 2
        assert len(model.arrange.simu_pile_coordinates) == 1
    
    def test_missing_coordinates(self, arrange_model_inputs: dict):
        """测试坐标数量不匹配的情况"""
        with pytest.raises(ValueError, match="桩布置信息不完整"):
            parse_arrange_text(arrange_model_inputs["missing_coordinates"])
    
    def test_missing_values(self, arrange_model_inputs: dict):
        """测试缺少必要值的情况"""
        with pytest.raises(ValueError, match="桩布置信息不完整"):
            parse_arrange_text(arrange_model_inputs["missing_values"])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
