import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the parent directory to the Python path so we can import the bcad_pile package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# 获取examples文件夹的路径
examples_dir = os.path.dirname(os.path.abspath(__file__))

from bcad_pile.core.computation import (
    analyze_pile_foundation,
    extract_visualization_data,
)


def create_test_dat_file(filename, config):
    """
    创建BCAD_PILE测试输入文件

    参数:
    filename (str): 输出文件名
    config (dict): 配置参数字典
    """
    with open(filename, "w") as f:
        # [contral] 块
        f.write("[contral]\n")
        f.write(f"{config['jctr']}\n")
        if config["jctr"] == 1:
            f.write(f"{config['nact']}\n")
            for i in range(config["nact"]):
                f.write(f"{config['axy'][i][0]} {config['axy'][i][1]}\n")
                f.write(" ".join(map(str, config["act"][i])) + "\n")
        elif config["jctr"] == 3:
            f.write(f"{config['ino']}\n")
        f.write("end;\n")

        # [arrange] 块
        f.write("[arrange]\n")
        f.write(f"{config['pnum']} {config['snum']}\n")
        for i in range(config["pnum"]):
            f.write(f"{config['pxy'][i][0]} {config['pxy'][i][1]}\n")
        if config["snum"] > 0:
            for i in range(config["snum"]):
                f.write(f"{config['sxy'][i][0]} {config['sxy'][i][1]}\n")
        f.write("end;\n")

        # [no_simu] 块
        f.write("[no_simu]\n")
        f.write(" ".join(map(str, config["kctr"])) + "\n")
        f.write("<0>\n")
        f.write(
            f"{config['ksh0']} {config['ksu0']} {' '.join(map(str, config['agl0']))}\n"
        )
        f.write(f"{config['nfr0']} {' '.join(map(str, config['hfr_dof_nsf0']))}\n")

        # 使用多行格式输出埋置段数据，每层土一行
        f.write(f"{config['nbl0']} ")
        for i in range(0, len(config["hbl_dob_pmt_pfi_nsg0"]), 5):
            f.write(
                " ".join(map(str, config["hbl_dob_pmt_pfi_nsg0"][i : i + 5])) + "\n"
            )

        f.write(f"{config['pmb0']} {config['peh0']} {config['pke0']}\n")
        f.write("end\n")

        # [simu_pe] 块如果需要
        f.write("[simu_pe]\n")
        if config["snum"] > 0:
            # 添加模拟桩的信息
            pass
        f.write("end;\n")


def parse_output_file(filename):
    """
    解析BCAD_PILE输出文件

    参数:
    filename (str): 输出文件名

    返回:
    dict: 包含解析结果的字典
    """
    results = {}
    with open(filename, "r") as f:
        lines = f.readlines()

    # 查找刚度矩阵的起始位置
    stiffness_start = None
    for i, line in enumerate(lines):
        if "Stiffness of the entire pile foundation" in line:
            stiffness_start = i + 2
            break

    if stiffness_start:
        # 提取6x6刚度矩阵
        stiffness_matrix = np.zeros((6, 6))
        for i in range(6):
            row_data = lines[stiffness_start + i].strip().split()
            for j in range(6):
                stiffness_matrix[i, j] = float(row_data[j])

        results["stiffness_matrix"] = stiffness_matrix

    return results


def visualize_pile_arrangement(config, output_file=None):
    """
    可视化桩的平面布置

    参数:
    config (dict): 配置参数字典
    output_file (str, optional): 输出文件路径
    """
    plt.figure(figsize=(10, 8))

    # 绘制非模拟桩
    x = [pos[0] for pos in config["pxy"]]
    y = [pos[1] for pos in config["pxy"]]
    plt.scatter(x, y, c="blue", marker="o", s=100, label="非模拟桩")

    # 绘制模拟桩(如果有)
    if config["snum"] > 0:
        x = [pos[0] for pos in config["sxy"]]
        y = [pos[1] for pos in config["sxy"]]
        plt.scatter(x, y, c="red", marker="x", s=100, label="模拟桩")

    plt.grid(True)
    plt.xlabel("X坐标 (m)")
    plt.ylabel("Y坐标 (m)")
    plt.title("桩基平面布置")
    plt.legend()
    plt.axis("equal")

    # 如果提供了输出文件路径，则保存到该路径
    if output_file:
        plt.savefig(output_file)
    else:
        plt.savefig("pile_arrangement.png")

    plt.close()


def visualize_stiffness_matrix(stiffness_matrix, output_file=None):
    """
    可视化刚度矩阵

    参数:
    stiffness_matrix (numpy.ndarray): 6x6刚度矩阵
    output_file (str, optional): 输出文件路径
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log10(np.abs(stiffness_matrix)), cmap="viridis")
    plt.colorbar(label="log10(绝对值)")

    # 添加数值标签
    for i in range(6):
        for j in range(6):
            plt.text(
                j,
                i,
                f"{stiffness_matrix[i, j]:.2e}",
                ha="center",
                va="center",
                color="w",
                fontsize=8,
            )

    plt.xticks(np.arange(6), ["Fx", "Fy", "Fz", "Mx", "My", "Mz"])
    plt.yticks(np.arange(6), ["Ux", "Uy", "Uz", "θx", "θy", "θz"])
    plt.title("桩基础刚度矩阵 (对数尺度)")

    # 如果提供了输出文件路径，则保存到该路径
    if output_file:
        plt.savefig(output_file)
    else:
        plt.savefig("stiffness_matrix.png")

    plt.close()


def run_test():
    """运行BCAD_PILE测试"""
    # T-1#.dat文件的基本配置
    config = {
        "jctr": 2,
        "pnum": 24,
        "snum": 0,
        "pxy": [
            [-5.0, 2.5],
            [-5.0, -2.5],
            [-5.0, 7.5],
            [-5.0, -7.5],
            [-5.0, 12.5],
            [-5.0, -12.5],
            [-5.0, 17.5],
            [-5.0, -17.5],
            [0, 2.5],
            [0, -2.5],
            [0, 7.5],
            [0, -7.5],
            [0, 12.5],
            [0, -12.5],
            [0, 17.5],
            [0, -17.5],
            [5.0, 2.5],
            [5.0, -2.5],
            [5.0, 7.5],
            [5.0, -7.5],
            [5.0, 12.5],
            [5.0, -12.5],
            [5.0, 17.5],
            [5.0, -17.5],
        ],
        "sxy": [],
        "kctr": [0] * 24,
        "ksh0": 0,
        "ksu0": 1,
        "agl0": [0.0, 0.0, 1.0],
        "nfr0": 0,
        "hfr_dof_nsf0": [],
        "nbl0": 13,
        "hbl_dob_pmt_pfi_nsg0": [
            13.4,
            2.0,
            5000,
            18.0,
            14,
            7.1,
            2.0,
            3000,
            13.7,
            8,
            1.5,
            2.0,
            5000,
            28.0,
            2,
            0.8,
            2.0,
            5000,
            15.5,
            1,
            5.5,
            2.0,
            10000,
            17.5,
            6,
            3.0,
            2.0,
            5000,
            15.5,
            3,
            11.0,
            2.0,
            5000,
            30.0,
            11,
            3.1,
            2.0,
            10000,
            17.5,
            4,
            7.0,
            2.0,
            10000,
            31.0,
            7,
            15.0,
            2.0,
            10000,
            17.0,
            15,
            3.8,
            2.0,
            10000,
            18.0,
            4,
            2.0,
            2.0,
            3000,
            12.5,
            2,
            16.8,
            2.0,
            5000,
            31.0,
            17,
        ],
        "pmb0": 5000,
        "peh0": 3e7,
        "pke0": 1,
    }

    # 创建测试文件，保存到examples文件夹
    test_file = os.path.join(examples_dir, "test.dat")
    create_test_dat_file(test_file, config)

    # 可视化桩布置，保存到examples文件夹
    visualize_pile_arrangement(
        config, os.path.join(examples_dir, "pile_arrangement.png")
    )

    # 运行分析
    results = analyze_pile_foundation(test_file)
    # 在实际应用中，需要运行BCAD_PILE程序并获取输出
    # subprocess.run(['bcad_pile.exe', 'test.dat'], capture_output=True)
    # 然后解析输出文件
    # results = parse_output_file("test.out")

    # 使用示例输出进行演示
    example_stiffness = np.array(
        [
            [6.326e07, 0.000e00, 0.000e00, 0.000e00, 2.692e08, 0.000e00],
            [0.000e00, 6.326e07, 0.000e00, -2.692e08, 0.000e00, 0.000e00],
            [0.000e00, 0.000e00, 1.896e08, 0.000e00, 0.000e00, 0.000e00],
            [0.000e00, -2.692e08, 0.000e00, 2.286e09, 0.000e00, 0.000e00],
            [2.692e08, 0.000e00, 0.000e00, 0.000e00, 2.286e09, 0.000e00],
            [0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00, 1.326e09],
        ]
    )
    # 可视化刚度矩阵，保存到examples文件夹
    visualize_stiffness_matrix(
        example_stiffness, os.path.join(examples_dir, "stiffness_matrix.png")
    )

    print("测试完成。生成了桩布置图和刚度矩阵可视化图。")


if __name__ == "__main__":
    run_test()
