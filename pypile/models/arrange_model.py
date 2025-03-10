#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
桩布置信息模型
"""

from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any
import re


# 桩坐标模型
class PileCoordinate(BaseModel):
    X: float = Field(..., description="桩的X坐标")
    Y: float = Field(..., description="桩的Y坐标")


# 桩布置模型
class ArrangeModel(BaseModel):
    PNUM: int = Field(..., description="非模拟桩的数量")
    SNUM: int = Field(..., description="模拟桩的数量")
    pile_coordinates: List[PileCoordinate] = Field(..., description="非模拟桩坐标列表")
    simu_pile_coordinates: List[PileCoordinate] = Field(
        ..., description="模拟桩坐标列表"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_coordinates_count(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证桩坐标数量是否符合PNUM和SNUM"""
        pile_coordinates = data.get("pile_coordinates", [])
        simu_pile_coordinates = data.get("simu_pile_coordinates", [])
        pnum = data.get("PNUM", 0)
        snum = data.get("SNUM", 0)

        if len(pile_coordinates) != pnum:
            raise ValueError(
                f"非模拟桩坐标数量({len(pile_coordinates)})与PNUM({pnum})不符"
            )

        if len(simu_pile_coordinates) != snum:
            raise ValueError(
                f"模拟桩坐标数量({len(simu_pile_coordinates)})与SNUM({snum})不符"
            )

        return data


# 桩布置信息解析模型
class ArrangeInfoModel(BaseModel):
    arrange: ArrangeModel

    @model_validator(mode="before")
    @classmethod
    def parse_input(cls, data: Any) -> Dict[str, Any]:
        # 解析输入文本
        raw_input = data.get("input_text", "")
        if not raw_input:
            raise ValueError("输入不能为空")

        # 如果[ARRANGE]标签不存在，直接在最前面添加[ARRANGE]
        if not re.search(r"\[ARRANGE\]", raw_input, re.IGNORECASE | re.DOTALL):
            raise ValueError("无法找到有效的布置块[ARRANGE]")

        # 使用正则表达式检查任意大小写的END
        if not re.search(r"END;", raw_input, re.IGNORECASE):
            raw_input += "\nEND;"

        # 将多个\n替换为\n
        raw_input = re.sub(r"\n+", "\n", raw_input)

        # 解析[ARRANGE] 到 END; 之间的内容
        arrange_pattern = r"\[ARRANGE\].*?(?:END;)"
        match = re.search(arrange_pattern, raw_input, re.IGNORECASE | re.DOTALL)
        if match:
            input_text = match.group(0)
        else:
            raise ValueError("无法找到有效的布置块")

        # 移除[ARRANGE]标签和END;，仅保留数据内容
        content = re.sub(
            r"\[ARRANGE\]|\s*(?:END;)", "", input_text, flags=re.IGNORECASE | re.DOTALL
        ).strip()

        # 提取所有的数字（整数和浮点数）
        values = list(map(float, re.findall(r"-?\d+\.?\d*", content)))
        if len(values) < 2:
            raise ValueError("桩布置信息不完整，至少需要PNUM和SNUM两个值")

        # 提取PNUM和SNUM
        pnum = int(values[0])
        snum = int(values[1])

        # 检查提取的数据是否足够构建坐标列表
        if len(values) < 2 + (pnum + snum) * 2:
            raise ValueError(
                f"桩布置信息不完整，需要{2 + (pnum + snum) * 2}个值，但只找到{len(values)}个"
            )

        # 构建非模拟桩坐标列表
        pile_coordinates = []
        for i in range(pnum):
            idx = 2 + i * 2
            pile_coordinates.append(PileCoordinate(X=values[idx], Y=values[idx + 1]))

        # 构建模拟桩坐标列表
        simu_pile_coordinates = []
        for i in range(snum):
            idx = 2 + pnum * 2 + i * 2
            simu_pile_coordinates.append(
                PileCoordinate(X=values[idx], Y=values[idx + 1])
            )

        # 构建ArrangeModel
        data["arrange"] = ArrangeModel(
            PNUM=pnum,
            SNUM=snum,
            pile_coordinates=pile_coordinates,
            simu_pile_coordinates=simu_pile_coordinates,
        )

        return data


# 工厂函数：根据输入文本创建适当的模型实例
def parse_arrange_text(input_text: str) -> ArrangeInfoModel:
    return ArrangeInfoModel(input_text=input_text)


if __name__ == "__main__":
    # 测试输入
    input_text = """
    [ARRANGE]
    2 1
    0.0 0.0
    5.0 0.0
    10.0 0.0
    END;
    """

    model = parse_arrange_text(input_text)
    print(f"PNUM: {model.arrange.PNUM}, SNUM: {model.arrange.SNUM}")
    print("非模拟桩坐标:")
    for i, coord in enumerate(model.arrange.pile_coordinates):
        print(f"  桩 {i + 1}: X={coord.X}, Y={coord.Y}")
    print("模拟桩坐标:")
    for i, coord in enumerate(model.arrange.simu_pile_coordinates):
        print(f"  桩 {i + 1}: X={coord.X}, Y={coord.Y}")
