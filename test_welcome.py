import sys
import os
import art
import random
import datetime

def print_welcome_message():
    """显示程序欢迎信息"""
    # 字体列表，用于动态展示不同字符形式
    font_list = ["wizard", "block", "alligator", "standard", "stellar", "slant", "doom"]
    # 框架符号列表
    frame_chars = ["+", "*", ".", "#", "=", "-"]
    
    # 随机选择字体和框架符号
    selected_font = random.choice(font_list)
    frame_char = random.choice(frame_chars)
    
    # 创建艺术文本
    art_text = art.text2art("PY PILE", font=selected_font)
    
    # 获取当前年份
    current_year = datetime.datetime.now().year
    version = "1.0"
    
    # 计算框架宽度（基于最长行）
    art_lines = art_text.split("\n")
    max_line_length = max(len(line) for line in art_lines)
    frame_width = max(max_line_length + 10, 80)  # 确保最小宽度为80
    
    # 创建上框架
    top_frame = art.line(length=frame_width, char=frame_char)
    
    # 创建装饰线
    decoration = art.decor("barcode1") + " " + art.decor("barcode1", reverse=True)
    
    # 构建完整的欢迎信息
    welcome_message = f"""
{top_frame}
{art_text}
{frame_char} {decoration} {frame_char}
{frame_char}{' ' * (frame_width - 2)}{frame_char}
{frame_char}{'版本号: ' + version + ' '*5 + '版权所有 © ' + str(current_year) + ' '*5 + '修改者: Lingyun Gou':^{frame_width-2}}{frame_char}
{frame_char}{' ' * (frame_width - 2)}{frame_char}
{frame_char}{'欢迎使用 pypile 程序！':^{frame_width-2}}{frame_char}
{frame_char}{' ' * (frame_width - 2)}{frame_char}
{frame_char}{'本程序旨在执行桥梁下部结构桩基的空间统计分析。':^{frame_width-2}}{frame_char}
{frame_char}{'如果您对本程序有任何疑问，请随时联系：':^{frame_width-2}}{frame_char}
{frame_char}{' ' * (frame_width - 2)}{frame_char}
{frame_char}{'CAD 研究组':^{frame_width-2}}{frame_char}
{frame_char}{'同济大学桥梁工程系':^{frame_width-2}}{frame_char}
{frame_char}{'中国上海四平路1239号':^{frame_width-2}}{frame_char}
{frame_char}{'邮编: 200092':^{frame_width-2}}{frame_char}
{frame_char}{' ' * (frame_width - 2)}{frame_char}
{top_frame}
"""
    
    # 输出欢迎信息
    print(welcome_message)
    
    # 保存到txt头文件
    try:
        with open("pypile_header.txt", "w", encoding="utf-8") as f:
            f.write(welcome_message)
    except Exception as e:
        print(f"无法保存头文件: {e}")

if __name__ == "__main__":
    print_welcome_message()
