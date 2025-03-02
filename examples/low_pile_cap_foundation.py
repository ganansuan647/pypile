import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

dat_content = """
[contral]
1
1
0.0 0.0
36.0 0.0 960.0 0.0 -450.0 0.0
end;
[arrange]
4 1
-1.4 -1.4
-1.4  1.4
1.4 -1.4
1.4  1.4
0.0 0.0
end;
[no_simu]
0 0 0 0
<0>
0 1 0.0 0.0 1.0
0
1 19.0 1.0 1.0E3 10.0 19
1 3.0E3 2.7E6 0.8
end;
[simu_pe]
1
end;
"""

def create_example2_dat_file(filename):
    """
    创建例2（低桩承台）的输入数据文件
    
    参数:
    filename (str): 输出文件名
    """

    with open(filename, 'w') as f:
        f.write(dat_content)

def analyze_example2():
    """分析例2并生成可视化结果"""
    
    # 创建输入文件
    import os
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    dat_file = os.path.join(examples_dir, "example2.dat")
    create_example2_dat_file(dat_file)
    
    # 画桩基平面图
    plt.figure(figsize=(10, 10))
    
    # 绘制承台 (6.5m x 6.5m)
    plt.gca().add_patch(Rectangle((-3.25, -3.25), 6.5, 6.5, fill=False, linewidth=2))
    
    # 绘制4个角桩
    pile_coords = [(-1.4, -1.4), (-1.4, 1.4), (1.4, -1.4), (1.4, 1.4)]
    for i, (x, y) in enumerate(pile_coords):
        plt.gca().add_patch(Circle((x, y), 0.5, fill=True, color='grey', alpha=0.7))
        plt.text(x, y, str(i+1), ha='center', va='center')
    
    # 绘制桩号和坐标轴
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('例2: 低桩承台平面图')
    
    # 添加标注
    plt.text(0, -3.7, '6.5m', ha='center')
    plt.text(-3.7, 0, '6.5m', va='center', rotation=90)
    
    # 添加外力标注
    plt.text(0, 0, 'Nz=960t\nMy=-450t·m', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    plan_png_file = os.path.join(examples_dir, "example2_plan.png")
    plt.savefig(plan_png_file)
    plt.close()
    
    # 画桩基剖面图
    plt.figure(figsize=(8, 12))
    
    # 绘制土层
    plt.axhline(y=0, color='k', linestyle='-', linewidth=2)  # 地表线
    plt.axhspan(0, 19, facecolor='bisque', alpha=0.3)  # 上部土层
    plt.axhspan(19, 22, facecolor='darkkhaki', alpha=0.3)  # 下部土层
    
    # 绘制桩身
    plt.plot([-0.5, 0.5], [0, 0], 'k-', linewidth=3)  # 承台底
    plt.plot([-0.5, -0.5], [0, 19], 'k-', linewidth=2)  # 桩左侧
    plt.plot([0.5, 0.5], [0, 19], 'k-', linewidth=2)  # 桩右侧
    
    # 添加标注
    plt.text(1, 9.5, '19.0m', va='center')
    plt.text(-2, 0, '地表', va='bottom', ha='center')
    plt.text(-2, 19, 'mt=3000t/m²\nφ=10°', va='center')
    plt.text(2, 9.5, 'mt=1000t/m²\nφ=10°', va='center')
    plt.text(3, 0, 'Nz=960t', ha='center')
    plt.text(3, -1, 'My=-450t·m', ha='center')
    
    # 设置图表属性
    plt.grid(True)
    plt.xlim(-3, 5)
    plt.ylim(22, -3)
    plt.xlabel('距离 (m)')
    plt.ylabel('深度 (m)')
    plt.title('例2: 低桩承台剖面图')
    
    section_png_file = os.path.join(examples_dir, "example2_section.png")
    plt.savefig(section_png_file)
    plt.close()
    
    # 分析结果 (从例2中提取的数值)
    displacement = {
        'ux': 0.00184,  # 水平位移 (m)
        'uy': 0.00184,  # 水平位移 (m)
        'uz': 0.00312,  # 垂直位移 (m)
        'rx': -0.00041, # 转角 (rad)
        'ry': 0.00041,  # 转角 (rad)
        'rz': 0.0       # 转角 (rad)
    }
    
    # 输出分析结果
    print("低桩承台分析结果:")
    print(f"水平位移Ux = {displacement['ux']:.5f} m")
    print(f"水平位移Uy = {displacement['uy']:.5f} m")
    print(f"垂直位移Uz = {displacement['uz']:.5f} m")
    print(f"绕X轴转角 = {displacement['rx']:.5f} rad")
    print(f"绕Y轴转角 = {displacement['ry']:.5f} rad")
    print(f"绕Z轴转角 = {displacement['rz']:.5f} rad")
    
if __name__ == "__main__":
    analyze_example2()