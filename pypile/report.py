
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
报告生成模块
"""

from typing import List, Dict, Union, Any, Optional
import numpy as np
from tabulate import tabulate
try:
    from models.pile_results_model import PileResult
except:
    from .models.pile_results_model import PileResult


def stiffness_matrix_report(stiffness_matrix: np.ndarray) -> str:
    """
    生成截面刚度矩阵的报告字符串
    
    Args:
        stiffness_matrix: 截面刚度矩阵, 应为6x6的numpy数组
        
    Returns:
        str: 格式化的报告字符串
    """
    if stiffness_matrix.shape != (6, 6):
        raise ValueError("刚度矩阵应为6x6的矩阵")
    
    # 设置刚度矩阵的行列标签
    dof_labels = ["Ux", "Uy", "Uz", "Rx", "Ry", "Rz"]
    
    # 创建标题
    report = "# 截面刚度矩阵报告\n\n"
    
    # 找出矩阵中的最大绝对值
    max_abs_value = np.max(np.abs(stiffness_matrix))
    # 设置阈值为最大值的1e-10
    threshold = max_abs_value * 1e-10
    
    # 添加矩阵内容
    table_data = []
    for i, row in enumerate(stiffness_matrix):
        formatted_row = [f"{dof_labels[i]}"]
        for val in row:
            # 如果值小于阈值，则显示为0
            if abs(val) < threshold:
                formatted_row.append("0")
            else:
                formatted_row.append(f"{val:.4e}")
        table_data.append(formatted_row)
    
    headers = ["DOF"] + dof_labels
    report += tabulate(table_data, headers=headers, tablefmt="grid") + "\n\n"
    
    # 添加单位说明
    report += "## 单位说明\n"
    report += "- 位移相关刚度: kN/m\n"
    report += "- 转角相关刚度: kN·m/rad\n"
    report += f"- 注: 比最大值小10个数量级的值已视为0 (阈值: {threshold:.4e})\n"
    
    return report

def pile_results_report(pile_results: Dict[int, PileResult]) -> str:
    """
    生成桩体验算结果的报告字符串
    
    Args:
        pile_results: 字典，键为桩号，值为PileResult对象
        
    Returns:
        str: 格式化的报告字符串
    """
    if not pile_results:
        return "没有桩体结果可供报告"
    
    report = f"# 桩体计算结果报告\n\n"
    
    # 汇总信息
    report += f"## 桩体汇总信息\n"
    report += f"- 总桩数: {len(pile_results)}\n"
    pile_numbers = sorted(pile_results.keys())
    report += f"- 桩号列表: {', '.join(str(p) for p in pile_numbers)}\n\n"
    
    # 为每根桩创建详细报告
    for pile_number in pile_numbers:
        pile_result = pile_results[pile_number]
        
        report += f"## 桩号 {pile_number} 详细信息\n\n"
        
        # 添加基本信息
        report += f"### 基本信息\n"
        report += f"- 桩号: {pile_number}\n"
        report += f"- 桩顶坐标: X={pile_result.coordinate[0]:.3f}m, Y={pile_result.coordinate[1]:.3f}m\n\n"
        
        # 添加桩顶位移和内力
        report += f"### 桩顶结果\n"
        top = pile_result.top_result
        
        # 位移表格
        displacement_data = [
            ["Ux (m)", "Uy (m)", "Uz (m)", "Rx (rad)", "Ry (rad)", "Rz (rad)"],
            [f"{top.UX:.6f}", f"{top.UY:.6f}", f"{top.UZ:.6f}", 
             f"{top.SX:.6f}", f"{top.SY:.6f}", f"{top.SZ:.6f}"]
        ]
        report += "#### Displacement\n"
        report += tabulate(displacement_data, tablefmt="grid", numalign="right", stralign="center") + "\n\n"
        
        # 内力表格
        force_data = [
            ["Nx (kN)", "Ny (kN)", "Nz (kN)", "Mx (kN·m)", "My (kN·m)", "Mz (kN·m)"],
            [f"{top.NX:.3f}", f"{top.NY:.3f}", f"{top.NZ:.3f}", 
             f"{top.MX:.3f}", f"{top.MY:.3f}", f"{top.MZ:.3f}"]
        ]
        report += "#### Force\n"
        report += tabulate(force_data, tablefmt="grid", numalign="right", stralign="center") + "\n\n"
        
        # 添加节点结果
        report += f"### 桩体节点结果\n"
        
        # 标记地面位置
        ground_level_text = f"地面位于节点索引 {pile_result.ground_level_index}"
        report += f"{ground_level_text}\n\n"
        
        # 创建节点数据表格
        node_data = []
        headers = ["   Z(m)   ","D (m)", "Ux (m)", "Uy (m)", "Rx (rad)", "Ry (rad)", 
                   "Nx (kN)", "Ny (kN)", "Nz (kN)", "Mx (kN·m)", "My (kN·m)",
                   "PSx (kN/m²)", "PSy (kN/m²)"]
        
        for i, node in enumerate(pile_result.nodes):
            # 为Z值添加足够的空格，确保列宽一致
            z_value = f"{node.Z:.2f}"
            if i == pile_result.ground_level_index:
                z_value += "(Ground)"
                
            row = [
                z_value,f"{node.D:.2f}", f"{node.UX:.6f}", f"{node.UY:.6f}", 
                f"{node.SX:.6f}", f"{node.SY:.6f}", 
                f"{node.NX:.3f}", f"{node.NY:.3f}", f"{node.NZ:.3f}", 
                f"{node.MX:.3f}", f"{node.MY:.3f}",
                f"{node.PSX:.3f}", f"{node.PSY:.3f}"
            ]
            
            node_data.append(row)
        
        report += tabulate(node_data, headers=headers, tablefmt="grid", numalign="right", stralign="center") + "\n\n"
        
        # 添加结论和评估
        report += "### 结论\n"
        
        # 计算最大值
        max_displacement_x = max([abs(node.UX) for node in pile_result.nodes])
        max_displacement_y = max([abs(node.UY) for node in pile_result.nodes])
        max_moment_x = max([abs(node.MX) for node in pile_result.nodes])
        max_moment_y = max([abs(node.MY) for node in pile_result.nodes])
        
        # 计算最大轴向力和剪力
        max_compression = max([node.NZ for node in pile_result.nodes])
        min_compression = min([node.NZ for node in pile_result.nodes])
        max_tension = -min_compression if min_compression < 0 else 0
        max_shear_x = max([abs(node.NX) for node in pile_result.nodes])
        max_shear_y = max([abs(node.NY) for node in pile_result.nodes])
        
        report += f"- {pile_number}桩最大水平位移(X): {max_displacement_x:.6f} m\n"
        report += f"- {pile_number}桩最大水平位移(Y): {max_displacement_y:.6f} m\n"
        report += f"- {pile_number}桩最大弯矩(X): {max_moment_x:.3f} kN·m\n"
        report += f"- {pile_number}桩最大弯矩(Y): {max_moment_y:.3f} kN·m\n"
        report += f"- {pile_number}桩最大轴压力: {max_compression:.3f} kN\n"
        report += f"- {pile_number}桩最大轴拉力: {max_tension:.3f} kN\n"
        report += f"- {pile_number}桩最大剪力(X): {max_shear_x:.3f} kN\n"
        report += f"- {pile_number}桩最大(Y): {max_shear_y:.3f} kN\n\n"
    
    # 添加总体结论
    report += "## 总体结论\n"
    
    # 计算整体最大值
    all_nodes = [node for result in pile_results.values() for node in result.nodes]
    max_disp_x_all = max([abs(node.UX) for node in all_nodes])
    max_disp_y_all = max([abs(node.UY) for node in all_nodes])
    max_moment_x_all = max([abs(node.MX) for node in all_nodes])
    max_moment_y_all = max([abs(node.MY) for node in all_nodes])
    max_compression_all = max([node.NZ for node in all_nodes])
    min_compression_all = min([node.NZ for node in all_nodes])
    max_tension_all = -min_compression_all if min_compression_all < 0 else 0
    max_shear_x_all = max([abs(node.NX) for node in all_nodes])
    max_shear_y_all = max([abs(node.NY) for node in all_nodes])
    
    report += f"- 全局最大水平位移(X): {max_disp_x_all:.6f} m\n"
    report += f"- 全局最大水平位移(Y): {max_disp_y_all:.6f} m\n"
    report += f"- 全局最大弯矩(X): {max_moment_x_all:.3f} kN·m\n"
    report += f"- 全局最大弯矩(Y): {max_moment_y_all:.3f} kN·m\n"
    report += f"- 全局最大轴压力: {max_compression_all:.3f} kN\n"
    report += f"- 全局最大轴拉力: {max_tension_all:.3f} kN\n"
    report += f"- 全局最大剪力(X): {max_shear_x_all:.3f} kN\n"
    report += f"- 全局最大剪力(Y): {max_shear_y_all:.3f} kN\n"
    
    return report

    def print_worst_pile_force(self) -> None:
        """输出最不利单桩内力结果到控制台和输出文件"""
        # 获取最不利内力
        worst_forces = self.get_worst_pile_force()
        
        # 获取沿桩身的最大弯矩和剪力
        pile_results = self.eforce()
        
        # 查找最大弯矩和剪力位置（沿桩身）
        max_moment_pile = -1
        max_moment_value = 0.0
        max_moment_location = 0.0
        max_moment_direction = ""
        
        max_shear_pile = -1
        max_shear_value = 0.0
        max_shear_location = 0.0
        max_shear_direction = ""
        
        for pile_num, result in pile_results.items():
            for node in result.nodes:
                # 计算合弯矩
                moment = (node.MX**2 + node.MY**2)**0.5
                if moment > max_moment_value:
                    max_moment_value = moment
                    max_moment_pile = pile_num
                    max_moment_location = node.Z
                    if abs(node.MX) > abs(node.MY):
                        max_moment_direction = "X"
                    else:
                        max_moment_direction = "Y"
                
                # 计算合剪力
                shear = (node.NX**2 + node.NY**2)**0.5
                if shear > max_shear_value:
                    max_shear_value = shear
                    max_shear_pile = pile_num
                    max_shear_location = node.Z
                    if abs(node.NX) > abs(node.NY):
                        max_shear_direction = "X"
                    else:
                        max_shear_direction = "Y"
        
        # 准备输出信息
        output_lines = [
            "\n最不利单桩内力计算结果：",
            "-" * 60,
            f"最大轴向力: {worst_forces['axial']['value']:10.2f} kN (桩号: {worst_forces['axial']['pile']+1})",
            f"最大X向侧力: {worst_forces['lateral_x']['value']:10.2f} kN (桩号: {worst_forces['lateral_x']['pile']+1})",
            f"最大Y向侧力: {worst_forces['lateral_y']['value']:10.2f} kN (桩号: {worst_forces['lateral_y']['pile']+1})",
            f"最大X向弯矩: {worst_forces['moment_x']['value']:10.2f} kN·m (桩号: {worst_forces['moment_x']['pile']+1})",
            f"最大Y向弯矩: {worst_forces['moment_y']['value']:10.2f} kN·m (桩号: {worst_forces['moment_y']['pile']+1})",
            f"最大扭矩: {worst_forces['torsion']['value']:10.2f} kN·m (桩号: {worst_forces['torsion']['pile']+1})",
            "-" * 60,
            f"沿桩身最大弯矩: {max_moment_value:10.2f} kN·m",
            f"  桩号: {max_moment_pile+1}",
            f"  位置: Z = {max_moment_location:6.2f} m",
            f"  主要方向: {max_moment_direction}",
            f"沿桩身最大剪力: {max_shear_value:10.2f} kN",
            f"  桩号: {max_shear_pile+1}",
            f"  位置: Z = {max_shear_location:6.2f} m",
            f"  主要方向: {max_shear_direction}",
            "-" * 60
        ]
        
        # 添加综合最不利桩信息
        combined_pile = worst_forces["combined"]["pile"]
        values = worst_forces["combined"]["values"]
        if combined_pile >= 0:
            output_lines.extend([
                f"综合最不利桩 (桩号: {combined_pile+1}):",
                f"  轴向力 NZ: {values[0]:10.2f} kN",
                f"  侧向力 NX: {values[1]:10.2f} kN",
                f"  侧向力 NY: {values[2]:10.2f} kN",
                f"  弯矩 MX: {values[3]:10.2f} kN·m",
                f"  弯矩 MY: {values[4]:10.2f} kN·m",
                f"  扭矩 MZ: {values[5]:10.2f} kN·m",
                "-" * 60
            ])
        
        # 输出到控制台
        for line in output_lines:
            print(line)
        
        # 如果输出文件已打开，也输出到文件
        if hasattr(self, 'output_file') and self.output_file:
            for line in output_lines:
                self.output_file.write(line + "\n")

def pile_group_report(pile_results: Dict[int, PileResult]) -> str:
    """
    生成桩群验算结果的综合报告
    
    Args:
        pile_results: 字典，键为桩号，值为PileResult对象
        
    Returns:
        str: 格式化的报告字符串
    """
    if not pile_results:
        return "没有桩体结果可供报告"
    
    report = "# 桩群计算结果综合报告\n\n"
    
    report += f"## 摘要\n"
    report += f"- 总桩数: {len(pile_results)}根\n\n"
    
    # 创建桩顶位移和内力摘要表格
    report += "## 桩顶位移和内力\n\n"
    
    # 位移和转角表格
    disp_data = []
    disp_headers = ["Pile No.", "X(m)", "Y(m)", 
                   "Ux(m)", "Uy(m)", "Uz(m)", 
                   "Sx(rad)", "Sy(rad)", "Sz(rad)"]
    
    for pile_number, result in pile_results.items():
        top = result.top_result
        row = [
            f"{pile_number}",
            f"{result.coordinate[0]:.3f}",
            f"{result.coordinate[1]:.3f}",
            f"{top.UX:.6f}",
            f"{top.UY:.6f}",
            f"{top.UZ:.6f}",
            f"{top.SX:.6f}",
            f"{top.SY:.6f}",
            f"{top.SZ:.6f}"
        ]
        disp_data.append(row)
    
    report += "### 位移和转角\n"
    report += tabulate(disp_data, headers=disp_headers, tablefmt="grid", numalign="right", stralign="center") + "\n\n"
    
    # 内力和弯矩表格
    force_data = []
    force_headers = ["Pile No.", "X(m)", "Y(m)", 
                    "Nx(kN)", "Ny(kN)", "Nz(kN)", 
                    "Mx(kN·m)", "My(kN·m)", "Mz(kN·m)"]
    
    for pile_number, result in pile_results.items():
        top = result.top_result
        row = [
            f"{pile_number}",
            f"{result.coordinate[0]:.3f}",
            f"{result.coordinate[1]:.3f}",
            f"{top.NX:.3f}",
            f"{top.NY:.3f}",
            f"{top.NZ:.3f}",
            f"{top.MX:.3f}",
            f"{top.MY:.3f}",
            f"{top.MZ:.3f}"
        ]
        force_data.append(row)
    
    report += "### 力和转角\n"
    report += tabulate(force_data, headers=force_headers, tablefmt="grid", numalign="right", stralign="center") + "\n\n"
    
    # 统计分析
    report += "## 统计分析\n\n"
    
    # 位移和转角统计
    report += "### 位移和转角统计\n"
    
    # 计算最大值、最小值和平均值
    max_ux = max(r.top_result.UX for r in pile_results.values())
    max_uy = max(r.top_result.UY for r in pile_results.values())
    max_uz = max(r.top_result.UZ for r in pile_results.values())
    max_sx = max(r.top_result.SX for r in pile_results.values())
    max_sy = max(r.top_result.SY for r in pile_results.values())
    max_sz = max(r.top_result.SZ for r in pile_results.values())
    
    min_ux = min(r.top_result.UX for r in pile_results.values())
    min_uy = min(r.top_result.UY for r in pile_results.values())
    min_uz = min(r.top_result.UZ for r in pile_results.values())
    min_sx = min(r.top_result.SX for r in pile_results.values())
    min_sy = min(r.top_result.SY for r in pile_results.values())
    min_sz = min(r.top_result.SZ for r in pile_results.values())
    
    avg_ux = sum(r.top_result.UX for r in pile_results.values()) / len(pile_results)
    avg_uy = sum(r.top_result.UY for r in pile_results.values()) / len(pile_results)
    avg_uz = sum(r.top_result.UZ for r in pile_results.values()) / len(pile_results)
    avg_sx = sum(r.top_result.SX for r in pile_results.values()) / len(pile_results)
    avg_sy = sum(r.top_result.SY for r in pile_results.values()) / len(pile_results)
    avg_sz = sum(r.top_result.SZ for r in pile_results.values()) / len(pile_results)
    
    disp_stat_data = [
        ["Max", f"{max_ux:.6f}", f"{max_uy:.6f}", f"{max_uz:.6f}", 
                  f"{max_sx:.6f}", f"{max_sy:.6f}", f"{max_sz:.6f}"],
        ["Min", f"{min_ux:.6f}", f"{min_uy:.6f}", f"{min_uz:.6f}", 
                  f"{min_sx:.6f}", f"{min_sy:.6f}", f"{min_sz:.6f}"],
        ["Avg", f"{avg_ux:.6f}", f"{avg_uy:.6f}", f"{avg_uz:.6f}", 
                  f"{avg_sx:.6f}", f"{avg_sy:.6f}", f"{avg_sz:.6f}"]
    ]
    
    disp_stat_headers = ["Statistics", "Ux(m)", "Uy(m)", "Uz(m)", 
                        "Sx(rad)", "Sy(rad)", "Sz(rad)"]
    
    report += tabulate(disp_stat_data, headers=disp_stat_headers, tablefmt="grid", numalign="right", stralign="center") + "\n\n"
    
    # 内力和弯矩统计
    report += "### 力和弯矩统计信息\n"
    
    max_nx = max(r.top_result.NX for r in pile_results.values())
    max_ny = max(r.top_result.NY for r in pile_results.values())
    max_nz = max(r.top_result.NZ for r in pile_results.values())
    max_mx = max(r.top_result.MX for r in pile_results.values())
    max_my = max(r.top_result.MY for r in pile_results.values())
    max_mz = max(r.top_result.MZ for r in pile_results.values())
    
    min_nx = min(r.top_result.NX for r in pile_results.values())
    min_ny = min(r.top_result.NY for r in pile_results.values())
    min_nz = min(r.top_result.NZ for r in pile_results.values())
    min_mx = min(r.top_result.MX for r in pile_results.values())
    min_my = min(r.top_result.MY for r in pile_results.values())
    min_mz = min(r.top_result.MZ for r in pile_results.values())
    
    avg_nx = sum(r.top_result.NX for r in pile_results.values()) / len(pile_results)
    avg_ny = sum(r.top_result.NY for r in pile_results.values()) / len(pile_results)
    avg_nz = sum(r.top_result.NZ for r in pile_results.values()) / len(pile_results)
    avg_mx = sum(r.top_result.MX for r in pile_results.values()) / len(pile_results)
    avg_my = sum(r.top_result.MY for r in pile_results.values()) / len(pile_results)
    avg_mz = sum(r.top_result.MZ for r in pile_results.values()) / len(pile_results)
    
    force_stat_data = [
        ["Max", f"{max_nx:.3f}", f"{max_ny:.3f}", f"{max_nz:.3f}", 
                  f"{max_mx:.3f}", f"{max_my:.3f}", f"{max_mz:.3f}"],
        ["Min", f"{min_nx:.3f}", f"{min_ny:.3f}", f"{min_nz:.3f}", 
                  f"{min_mx:.3f}", f"{min_my:.3f}", f"{min_mz:.3f}"],
        ["Avg", f"{avg_nx:.3f}", f"{avg_ny:.3f}", f"{avg_nz:.3f}", 
                  f"{avg_mx:.3f}", f"{avg_my:.3f}", f"{avg_mz:.3f}"]
    ]
    
    force_stat_headers = ["Statistics", "Nx(kN)", "Ny(kN)", "Nz(kN)", 
                         "Mx(kN·m)", "My(kN·m)", "Mz(kN·m)"]
    
    report += tabulate(force_stat_data, headers=force_stat_headers, tablefmt="grid", numalign="right", stralign="center") + "\n\n"
    
    return report

def worst_pile_report(worst_forces: Dict) -> str:
    """生成最不利单桩报告"""
    # 获取沿桩身的最大弯矩和剪力
    pile_results = worst_forces["pile_results"]
    
    # 查找最大弯矩和剪力位置（沿桩身）
    max_moment_pile = -1
    max_moment_value = 0.0
    max_moment_location = 0.0
    max_moment_direction = ""
    
    max_shear_pile = -1
    max_shear_value = 0.0
    max_shear_location = 0.0
    max_shear_direction = ""
    
    for pile_num, result in pile_results.items():
        for node in result.nodes:
            # 计算合弯矩
            moment = (node.MX**2 + node.MY**2)**0.5
            if moment > max_moment_value:
                max_moment_value = moment
                max_moment_pile = pile_num
                max_moment_location = node.Z
                if abs(node.MX) > abs(node.MY):
                    max_moment_direction = "X"
                else:
                    max_moment_direction = "Y"
            
            # 计算合剪力
            shear = (node.NX**2 + node.NY**2)**0.5
            if shear > max_shear_value:
                max_shear_value = shear
                max_shear_pile = pile_num
                max_shear_location = node.Z
                if abs(node.NX) > abs(node.NY):
                    max_shear_direction = "X"
                else:
                    max_shear_direction = "Y"
    
    # 准备输出信息
    output_lines = [
        "\n最不利单桩内力计算结果：",
        "-" * 60,
        f"最大轴向力: {worst_forces['axial']['value']:10.2f} kN (桩号: {worst_forces['axial']['pile']+1})",
        f"最大X向侧力: {worst_forces['lateral_x']['value']:10.2f} kN (桩号: {worst_forces['lateral_x']['pile']+1})",
        f"最大Y向侧力: {worst_forces['lateral_y']['value']:10.2f} kN (桩号: {worst_forces['lateral_y']['pile']+1})",
        f"最大X向弯矩: {worst_forces['moment_x']['value']:10.2f} kN·m (桩号: {worst_forces['moment_x']['pile']+1})",
        f"最大Y向弯矩: {worst_forces['moment_y']['value']:10.2f} kN·m (桩号: {worst_forces['moment_y']['pile']+1})",
        f"最大扭矩: {worst_forces['torsion']['value']:10.2f} kN·m (桩号: {worst_forces['torsion']['pile']+1})",
        "-" * 60,
        f"沿桩身最大弯矩: {max_moment_value:10.2f} kN·m",
        f"  桩号: {max_moment_pile+1}",
        f"  位置: Z = {max_moment_location:6.2f} m",
        f"  主要方向: {max_moment_direction}",
        f"沿桩身最大剪力: {max_shear_value:10.2f} kN",
        f"  桩号: {max_shear_pile+1}",
        f"  位置: Z = {max_shear_location:6.2f} m",
        f"  主要方向: {max_shear_direction}",
        "-" * 60
    ]
    
    # 添加综合最不利桩信息
    combined_pile = worst_forces["combined"]["pile"]
    values = worst_forces["combined"]["values"]
    if combined_pile >= 0:
        output_lines.extend([
            f"综合最不利桩 (桩号: {combined_pile+1}):",
            f"  轴向力 NZ: {values[0]:10.2f} kN",
            f"  侧向力 NX: {values[1]:10.2f} kN",
            f"  侧向力 NY: {values[2]:10.2f} kN",
            f"  弯矩 MX: {values[3]:10.2f} kN·m",
            f"  弯矩 MY: {values[4]:10.2f} kN·m",
            f"  扭矩 MZ: {values[5]:10.2f} kN·m",
            "-" * 60
        ])
    
    return "\n".join(output_lines)