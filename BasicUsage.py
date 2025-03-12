from pypile import PileManager
from pathlib import Path
import numpy as np

def use_cli():
    from pypile import pypile
    pypile()

if __name__ == "__main__":
    # 在python脚本中使用
    pile = PileManager()
    pile.read_dat(Path("./tests/Test-1-2.dat"))
    
    np.set_printoptions(linewidth=200, precision=2, suppress=True)
    # 使用pile.K查看基础刚度
    print(f"Pile stiffness matrix K:\n{pile.K}")
    # 使用pile.K_pile(桩号)查看指定桩的刚度
    # ino:int = 5
    # print(f"Pile {ino} stiffness matrix:\n{pile.K_pile(ino)}")
    
    # 无需在意是否已经在dat文件中设置了荷载，可以直接输入
    force = np.array([22927.01, 0, 40702.94, 0.0, 320150.23, 0])
    np.set_printoptions(linewidth=200, precision=4, suppress=True)
    # 打印得到的位移
    print(f"Cap displacement:\n{pile.disp_cap(force)}")         #承台位移
    # print(f"Pile displacement:\n{pile.disp_piles(force)}")      #各桩桩顶位移

    # 计算基础反力，得到一个Dict，key是桩号(int)，value是PileResult对象
    pile_results = pile.eforce(force)
    # pileResult对象有以下字段：
    print("\nPileResult object fields:")
    for field, info in pile_results[0].model_fields.items():
        print(f"{field}: {info.description}")

    print("\n")

    # 可以遍历提取对应结果
    # for pile_id,result in pile_results.items():
    #     reaction = "NZ"   # 对应轴力
    #     # 各桩顶轴力
    #     print(f"Pile {pile_id} at {result.coordinate}, \t{result.top_result.model_fields[reaction].description}:{getattr(result.top_result, reaction):.4e}")

    # for node in pile_results[0].nodes:
    #     reaction = "MY"   # 对应弯矩
    #     # 第1根桩弯矩沿桩身的分布
    #     print(f"z:{node.Z:.1f}m {node.model_fields[reaction].description}:{getattr(node, reaction):.4e}")

    # 还可以查看刚度矩阵报告，通过参数output_file指定输出文件
    pile.stiffness_report()
    # 群桩基础报告
    pile.pile_group_report()
    # 最不利单桩报告
    pile.worst_pile_report()

    # 也可以调用pile.worst_pile_force属性直接拿到最不利单桩结果
    print(pile.worst_pile_force)

    # 在终端中调用
    # use_cli()
    