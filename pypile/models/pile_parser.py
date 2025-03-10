from pydantic_core.core_schema import model_field
from .arrange_model import parse_arrange_text, ArrangeModel
from .simu_pile_model import parse_simu_pile_text, SimuPileModel
from .no_simu_model import parse_no_simu_text, NoSimuModel
from .control_model import parse_control_text, ControlModel

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Any
import re


class PileModel(BaseModel):
    name: str = Field(description="桩基础名称")
    control: ControlModel = Field(description="计算控制参数")
    arrange: ArrangeModel = Field(description="桩布置信息")
    no_simu: NoSimuModel = Field(description="非模拟桩信息")
    simu_pile: SimuPileModel = Field(description="模拟桩信息")

    @model_validator(mode="before")
    @classmethod
    def parse_input(cls, data: Any) -> Dict[str, Any]:
        # 解析输入文本
        raw_input = data.get("input_text", "")
        if not raw_input:
            raise ValueError("输入不能为空")

        # 获取name字段，如果存在的话
        name = data.get("name", "")

        # 如果[CONTRAL] 或 [CONTROL]标签不存在,报错
        if not re.search(
            r"\[\s*(?:CONTROL|CONTRAL)\s*\]", raw_input, re.IGNORECASE | re.DOTALL
        ):
            raise ValueError("无法找到有效的控制信息标签[Control]")

        # 检查并确保各个控制块之间有END;标记
        control_tags = [
            r"\[\s*(?:CONTROL|CONTRAL)\s*\]",
            r"\[\s*(?:ARRANGE|ARRAGE)\s*\]",
            r"\[\s*(?:NO_SIMU|NOSIMU)\s*\]",
            r"\[\s*(?:SIMU_PILE|SIMUPILE|simu_pe)\s*\]",
        ]

        # 添加文件末尾的处理
        control_tags.append("$")

        # 查找所有控制标签的位置
        tag_positions = []
        for tag in control_tags[:-1]:  # 不包括文件末尾的标记
            for match in re.finditer(tag, raw_input, re.IGNORECASE | re.DOTALL):
                tag_positions.append((match.start(), match.group(0)))

        # 按位置排序
        tag_positions.sort()

        # 处理后的输入文本
        processed_input = raw_input

        # 检查每个控制块是否以END;结束，如果没有则添加
        offset = 0  # 用于调整位置偏移
        for i in range(len(tag_positions)):
            current_pos, current_tag = tag_positions[i]

            # 确定当前块的结束位置
            end_pos = (
                len(processed_input)
                if i == len(tag_positions) - 1
                else tag_positions[i + 1][0]
            )

            # 提取当前块的文本
            current_block = processed_input[current_pos + offset : end_pos + offset]

            # 检查当前块是否包含END;
            if not re.search(r"END;", current_block, re.IGNORECASE):
                # 在块的末尾插入END;
                insert_pos = end_pos + offset
                processed_input = (
                    processed_input[:insert_pos]
                    + "\nEND;\n"
                    + processed_input[insert_pos:]
                )
                offset += 6  # 调整偏移量（"\nEND;\n"的长度）

        # 将处理后的文本赋回给data
        data["input_text"] = processed_input

        # 解析控制块
        control = parse_control_text(processed_input)

        # 解析布置块
        arrange = parse_arrange_text(processed_input)

        # 解析非模拟桩块
        no_simu = parse_no_simu_text(processed_input)

        # 解析模拟桩块
        simu_pile = parse_simu_pile_text(processed_input)

        # 返回解析后的model_field，包括name字段
        return {
            "name": name,
            "control": control.control,
            "arrange": arrange.arrange,
            "no_simu": no_simu.no_simu,
            "simu_pile": simu_pile.simu_pile,
        }


def parse_pile_text(input_text: str) -> PileModel:
    return PileModel(input_text=input_text)


if __name__ == "__main__":
    test_input = """
    [CONTRAL]
1
1
0 0
19545.36 0 31856.21 0 -289720.41 0
END;
[arrange]
24 0
-4.5 3.625
-4.5 -3.625
-4.5 8.125
-4.5 -8.125
-4.5 12.625
-4.5 -12.625
-4.5 17.125
-4.5 -17.125
0 3.625
0 -3.625
0 8.125
0 -8.125
0 12.625
0 -12.625
0 17.125
0 -17.125
4.5 3.625
4.5 -3.625
4.5 8.125
4.5 -8.125
4.5 12.625
4.5 -12.625
4.5 17.125
4.5 -17.125
end;
[no_simu]
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
<0>
0 1 0 0 1
1 4.1 1.8 5
7 17.5 1.8 3000 13.7 18
 25.7 1.8 5000 15.5 26
 14.1 1.8 5000 30.0 15
 1.0 1.8 5000 17.0 1
 5.9 1.8 10000 17.0 6
 2.5 1.8 5000 17.0 3
 14.2 1.8 5000 31.0 15
5000 3e7 1
end;
[simu_pe]
0
<0>
0 1 0 0 1
1 4.1 1.8 5
1 17.5 1.8 3000 13.7 18
5000 3e7 1
end;
    """

    # 正确传递参数 - 将测试输入文本放入字典中，并使用'input_text'作为键
    model = PileModel(name="pile1", input_text=test_input)
    print(model)
