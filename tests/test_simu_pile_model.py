#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模拟桩信息模型的pytest测试文件
"""

from typing import Dict

import pytest
from pydantic import ValidationError

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pypile.models.simu_pile_model import SimuPileInfoModel, parse_simu_pile_text


@pytest.fixture
def simu_pile_model_inputs() -> Dict:
    """提供不同测试场景的输入数据"""
    return {
        "standard": """
        [SIMUPILE]
        4 5
        <4>
        0 2 0.0 0.0 1.0
        1 5.0 1.0 2
        2 10.0 1.0 30000.0 30.0 3 15.0 1.0 28000.0 28.0 3
        25000.0 30000000.0 1.0
        <5>
        1 3 0.0 0.0 1.0
        0
        1 10.0 1.0 35000.0 30.0 2
        30000.0 32000000.0 1.0
        END;
        """,
        "single_type": """
        [simu_pe]
        5 5 5
        <5>
        0 2 0.0 0.0 1.0
        1 5.0 1.0 2
        1 10.0 1.0 30000.0 30.0 3
        25000.0 30000000.0 1.0
        END;
        """,
        "no_above_ground": """
        [SIMUPILE]
        6
        <6>
        0 2 0.0 0.0 1.0
        0
        1 10.0 1.0 30000.0 30.0 3
        25000.0 30000000.0 1.0
        END;
        """,
        "invalid_ksh": """
        [SIMUPILE]
        7
        <7>
        2 2 0.0 0.0 1.0
        1 5.0 1.0 2
        1 10.0 1.0 30000.0 30.0 3
        25000.0 30000000.0 1.0
        END;
        """,
        "invalid_ksu": """
        [simu_pe]
        8
        <8>
        0 5 0.0 0.0 1.0
        1 5.0 1.0 2
        1 10.0 1.0 30000.0 30.0 3
        25000.0 30000000.0 1.0
        END;
        """,
        "invalid_agl": """
        [SIMUPILE]
        9
        <9>
        0 2 0.0 0.0 0.5
        1 5.0 1.0 2
        1 10.0 1.0 30000.0 30.0 3
        25000.0 30000000.0 1.0
        END;
        """,
        "missing_tag": """
        4 5
        <4>
        0 2 0.0 0.0 1.0
        1 5.0 1.0 2
        2 10.0 1.0 30000.0 30.0 3 15.0 1.0 28000.0 28.0 3
        25000.0 30000000.0 1.0
        <5>
        0 2 0.0 0.0 0.5
        1 5.0 1.0 2
        1 10.0 1.0 30000.0 30.0 3
        25000.0 30000000.0 1.0
        END;
        """,
        "missing_end": """
        [simu_pe]
        4 4 4
        <4>
        0 2 0.0 0.0 1.0
        1 5.0 1.0 2
        2 10.0 1.0 30000.0 30.0 3 15.0 1.0 28000.0 28.0 3
        25000.0 30000000.0 1.0
        """,
        "invalid_pfi": """
        [SIMUPILE]
        10
        <10>
        0 2 0.0 0.0 1.0
        1 5.0 1.0 2
        1 10.0 1.0 30000.0 95.0 3
        25000.0 30000000.0 1.0
        END;
        """,
        "inconsistent_type_id": """
        [simu_pe]
        11
        <12>
        0 2 0.0 0.0 1.0
        1 5.0 1.0 2
        1 10.0 1.0 30000.0 30.0 3
        25000.0 30000000.0 1.0
        END;
        """,
    }


class TestSimuPileModel:
    """模拟桩信息模型的测试类"""

    def test_standard_simu_pile(self, simu_pile_model_inputs: Dict):
        """测试标准的模拟桩信息"""
        model = parse_simu_pile_text(simu_pile_model_inputs["standard"])

        # 验证KSCTR列表
        assert model.simu_pile.KSCTR == [4, 5]

        # 验证桩类型数量
        assert len(model.simu_pile.pile_types) == 2

        # 验证第一种桩类型信息
        pile_type_4 = model.simu_pile.pile_types[4]
        assert pile_type_4.KSH == 0
        assert pile_type_4.KSU == 2
        assert pile_type_4.AGL == [0.0, 0.0, 1.0]
        assert pile_type_4.NFR == 1
        assert len(pile_type_4.above_ground_sections) == 1
        assert pile_type_4.NBL == 2
        assert len(pile_type_4.below_ground_sections) == 2
        assert pile_type_4.PMB == 25000.0
        assert pile_type_4.PEH == 30000000.0
        assert pile_type_4.PKE == 1.0

        # 验证第一个地上桩段
        above_section = pile_type_4.above_ground_sections[0]
        assert above_section.HFR == 5.0
        assert above_section.DOF == 1.0
        assert above_section.NSF == 2

        # 验证第一个地下桩段
        below_section = pile_type_4.below_ground_sections[0]
        assert below_section.HBL == 10.0
        assert below_section.DOB == 1.0
        assert below_section.PMT == 30000.0
        assert below_section.PFI == 30.0
        assert below_section.NSG == 3

        # 验证第二种桩类型信息
        pile_type_5 = model.simu_pile.pile_types[5]
        assert pile_type_5.KSH == 1
        assert pile_type_5.KSU == 3
        assert pile_type_5.NFR == 0
        assert pile_type_5.NBL == 1

    def test_single_type(self, simu_pile_model_inputs: Dict):
        """测试单一桩类型的情况"""
        model = parse_simu_pile_text(simu_pile_model_inputs["single_type"])
        assert model.simu_pile.KSCTR == [5, 5, 5]
        assert len(model.simu_pile.pile_types) == 1

        # 验证桩类型信息
        pile_type = model.simu_pile.pile_types[5]
        assert pile_type.NFR == 1
        assert pile_type.NBL == 1

    def test_no_above_ground(self, simu_pile_model_inputs: Dict):
        """测试无地上桩段的情况"""
        model = parse_simu_pile_text(simu_pile_model_inputs["no_above_ground"])
        assert model.simu_pile.KSCTR == [6]
        assert len(model.simu_pile.pile_types) == 1

        # 验证桩类型信息
        pile_type = model.simu_pile.pile_types[6]
        assert pile_type.NFR == 0
        assert len(pile_type.above_ground_sections) == 0
        assert pile_type.NBL == 1
        assert len(pile_type.below_ground_sections) == 1

    def test_invalid_ksh(self, simu_pile_model_inputs: Dict):
        """测试无效的KSH值"""
        with pytest.raises(ValidationError) as excinfo:
            parse_simu_pile_text(simu_pile_model_inputs["invalid_ksh"])
        assert "KSH必须是0(圆截面)或1(方截面)" in str(excinfo.value)

    def test_invalid_ksu(self, simu_pile_model_inputs: Dict):
        """测试无效的KSU值"""
        with pytest.raises(ValidationError) as excinfo:
            parse_simu_pile_text(simu_pile_model_inputs["invalid_ksu"])
        assert "KSU必须是1(摩擦桩), 2(摩擦端承桩), 3(端承桩)或4(其他边界条件)" in str(
            excinfo.value
        )

    def test_invalid_agl(self, simu_pile_model_inputs: Dict):
        """测试无效的AGL值"""
        with pytest.raises(ValidationError) as excinfo:
            parse_simu_pile_text(simu_pile_model_inputs["invalid_agl"])
        assert "方向余弦的平方和必须等于1" in str(excinfo.value)

    def test_missing_tag(self, simu_pile_model_inputs: Dict):
        """测试缺少[SIMUPILE]标签的情况"""
        # 缺少标签应该报错
        with pytest.raises(ValidationError) as excinfo:
            parse_simu_pile_text(simu_pile_model_inputs["missing_tag"])
        assert "无法找到有效的模拟桩信息标签[Simu Pile]或[simu_pe]" in str(
            excinfo.value
        )

    def test_missing_end(self, simu_pile_model_inputs: Dict):
        """测试缺少END;标签的情况"""
        # 缺少END;应该被自动添加
        model = parse_simu_pile_text(simu_pile_model_inputs["missing_end"])
        assert model.simu_pile.KSCTR == [4, 4, 4]

    def test_invalid_pfi(self, simu_pile_model_inputs: Dict):
        """测试无效的土内摩擦角值"""
        with pytest.raises(ValidationError) as excinfo:
            parse_simu_pile_text(simu_pile_model_inputs["invalid_pfi"])
        assert "土内摩擦角必须在0-90度范围内" in str(excinfo.value)

    def test_inconsistent_type_id(self, simu_pile_model_inputs: Dict):
        """测试桩类型ID与KSCTR不匹配的情况"""
        with pytest.raises(ValidationError) as excinfo:
            parse_simu_pile_text(simu_pile_model_inputs["inconsistent_type_id"])
        assert (
            "KSCTR(模拟桩类型标识)中存在值:11,但并未找到对应的土层信息，请检查!"
            in str(excinfo.value)
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
