#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyPile Python版本
Created by: Lingyun Gou, Dept. of Bridge Engr.,Tongji University
Date: 2025-03-09
"""

import numpy as np
import math
from pathlib import Path
from loguru import logger

from models import PileModel

try:
    from . import __version__
except ImportError:
    __version__ = "Debug"

class PileManager:
    def __init__(self, debug: bool = False):
        self.debug: bool = debug
        # 配置日志系统
        if debug:
            # debug 模式下输出日志文件和控制台信息
            logger.add("pypile.log", level="DEBUG")
            logger.info("Debug mode enabled, detailed logs will be written to pypile.log")
        else:
            # 非 debug 模式下只保留控制台 WARNING 及以上级别的信息
            logger.remove()  # 移除默认处理器
            logger.add(lambda msg: print(msg, end=""), level="WARNING")  # 只在控制台输出警告及以上级别
        # 输入输出文件
        self.input_file = None
        self.output_file = None
        self.pos_file = None
        
        print(self.welcome_message)

    @property
    def welcome_message(self):
        import random
        import art
        import datetime
        import textwrap
        
        def generate_ascii_art(text, fonts):
            font_list = fonts
            font = random.choice(font_list)
            ascii_art = art.text2art(text, font=font)
            return ascii_art

        def create_bordered_ascii(ascii_art, version):
            lines = ascii_art.split('\n')
            width = max(len(line) for line in lines)
            height = len(lines)
        
            horizontal_border = '-' * (width + 3)
            vertical_padding = ' ' * 1
            
            bordered = [horizontal_border]
            for line in lines:
                bordered.append(f"|{vertical_padding}{line:<{width}}{vertical_padding}|")
            
            # 添加版本信息、版权声明和修改人信息到最后一行
            copyright_info = f"Version:{version}, {datetime.datetime.now().year}"
            bordered.append(f"|{vertical_padding}{' ':<{width-len(copyright_info)}}{copyright_info}{vertical_padding}|")
            author_info = "By: Lingyun Gou"
            bordered.append(f"|{vertical_padding}{' ':<{width-len(author_info)}}{author_info}{vertical_padding}|")
            bordered.append(horizontal_border)
            
            welcome_text = "Welcome to use the pypile program !!"
            
            # 使用textwrap模块自动换行
            info_text1 = "This program is aimed to execute spatial statical analysis of pile foundations of bridge substructures."
            info_text2 = "If you have any questions about this program, please do not hesitate to create an issue on GitHub or write to:"
            
            wrapped_info_text1 = textwrap.fill(info_text1, width=width+3)
            wrapped_info_text2 = textwrap.fill(info_text2, width=width+3)
            
            # 联系信息右对齐
            contact_info = [
                "CAD Research Group",
                "Dept. of Bridge Engr.",
                "Tongji University",
                "1239 Sipin Road", 
                "Shanghai 200092",
                "P.R. of China"
            ]
            
            # 使用:<进行右对齐
            right_aligned_contact = [f"{line:>{width+3}}" for line in contact_info]
            
            # 添加边框外的欢迎信息和联系信息
            result = '\n'.join(bordered)
            result = welcome_text + '\n' + result + '\n' + wrapped_info_text1 + '\n\n' + wrapped_info_text2 + '\n\n' + '\n'.join(right_aligned_contact)
            
            return result

        fonts = ["epic", "doh", "colossal",'block2', "basic", "alpha", "3d_diagonal", "3-d", "1943", "doom", "fire_font", "ghost", "graceful", "graffiti", "impossible", "lean", "letters", "modular", "rounded", "soft", "starwars", "speed2", "sub-zero", "twisted", "varsity"]

        ascii_art = generate_ascii_art("PY PILE", fonts)
        welcome_message = create_bordered_ascii(ascii_art, __version__)
        return welcome_message

    def calculate_total_force(self, force_points):
        """计算外部荷载的合力"""
        force = np.zeros(6, dtype=float)
        
        for point in force_points:
            x, y = point[:2]
            local_force = point[2:]
            transformation_matrix = self.tmatx(x, y)
            global_force = np.dot(transformation_matrix.T, local_force)
            force += global_force
        
        return force

    def init_parameters(self, Pile: PileModel):
        """初始化参数"""
        self.Pile = Pile
        logger.debug(f"Initializing parameters for {Pile.name}...")
        # 初始化参数
        self.pnum = Pile.arrange.PNUM   # 非模拟桩数量，来自 arrange模块
        self.snum = Pile.arrange.SNUM   # 模拟桩数量，来自 arrange模块
        self.N_max_pile = self.pnum + self.snum
        self.N_max_layer = max(pile_type.NFR+pile_type.NBL for pile_type in Pile.no_simu.pile_types.values())
        self.N_max_calc_points = max(
            sum(layer.NSF for layer in pile_type.above_ground_sections) +
            sum(layer.NSG for layer in pile_type.below_ground_sections)
            for pile_type in Pile.no_simu.pile_types.values()
        ) + 1   # 计算点数
        
        # 非模拟桩信息
        self.ksh = np.zeros(self.N_max_pile, dtype=int)         # 桩断面形状(0-圆形,1-方形)
        self.ksu = np.zeros(self.N_max_pile, dtype=int)         # 桩底约束条件
        self.agl = np.zeros((self.N_max_pile, 3), dtype=float)  # 桩的倾斜方向余弦
        self.nfr = np.zeros(self.N_max_pile, dtype=int)         # 桩地上段段数
        self.hfr = np.zeros((self.N_max_pile, self.N_max_layer), dtype=float) # 桩地上段每段高度
        self.dof = np.zeros((self.N_max_pile, self.N_max_layer), dtype=float) # 桩地上段每段直径
        self.nsf = np.zeros((self.N_max_pile, self.N_max_layer), dtype=int)   # 桩地上段计算分段数
        self.nbl = np.zeros(self.N_max_pile, dtype=int)         # 桩地下段段数
        self.hbl = np.zeros((self.N_max_pile, self.N_max_layer), dtype=float) # 桩地下段每段高度
        self.dob = np.zeros((self.N_max_pile, self.N_max_layer), dtype=float) # 桩地下段每段直径
        self.pmt = np.zeros((self.N_max_pile, self.N_max_layer), dtype=float) # 桩地下段每段地基反力系数
        self.pfi = np.zeros((self.N_max_pile, self.N_max_layer), dtype=float) # 桩地下段每段摩擦角
        self.nsg = np.zeros((self.N_max_pile, self.N_max_layer), dtype=int)   # 桩地下段计算分段数
        self.pmb = np.zeros(self.N_max_pile, dtype=float)       # 桩端土抗力系数
        self.peh = np.zeros(self.N_max_pile, dtype=float)       # 桩材弹性模量
        self.pke = np.zeros(self.N_max_pile, dtype=float)       # 桩材剪切模量与弹性模量比

        # 模拟桩信息
        self.sxy = np.zeros((self.snum, 2), dtype=float)    # 模拟桩坐标

        # control
        self.jctr = Pile.control.JCTR
        if self.jctr == 1:
            # JCTR = 1：执行完整分析
            # 计算总荷载
            self.force = self.calculate_total_force(Pile.control.force_points)
        elif self.jctr == 2:
            # JCTR = 2：仅计算整个桩基础的刚度矩阵
            pass
        elif self.jctr == 3:
            # JCTR = 3：仅计算指定单桩的刚度矩阵
            self.ino = Pile.control.ino
        
        # arrange - 设置桩的坐标信息
        self.pxy = np.array([[coord.X, coord.Y] for coord in Pile.arrange.pile_coordinates])
        if self.snum > 0:
            self.sxy[:self.snum, :] = np.array([[coord.X, coord.Y] for coord in Pile.arrange.simu_pile_coordinates])
        
        # no_simu - 设置非模拟桩信息
        # 读取KCTR
        self.kctr = np.array(Pile.no_simu.KCTR, dtype=int)
        
        all_pile_ids = Pile.no_simu.pile_types.keys()        
        # 为每个桩填充信息
        for k in range(self.pnum):
            pile_type_id = self.kctr[k]
            # 获取对应类型的桩信息
            if pile_type_id in all_pile_ids:
                pile_info = Pile.no_simu.pile_types[pile_type_id]
                
                # 设置桩基本信息
                self.ksh[k] = pile_info.KSH
                self.ksu[k] = pile_info.KSU
                self.agl[k, :] = np.array(pile_info.AGL)
                
                # 设置地上段信息
                self.nfr[k] = pile_info.NFR
                for i in range(pile_info.NFR):
                    section = pile_info.above_ground_sections[i]
                    self.hfr[k, i] = section.HFR
                    self.dof[k, i] = section.DOF
                    self.nsf[k, i] = section.NSF
                
                # 设置地下段信息
                self.nbl[k] = pile_info.NBL
                for i in range(pile_info.NBL):
                    section = pile_info.below_ground_sections[i]
                    self.hbl[k, i] = section.HBL
                    self.dob[k, i] = section.DOB
                    self.pmt[k, i] = section.PMT
                    self.pfi[k, i] = section.PFI
                    self.nsg[k, i] = section.NSG
                
                # 设置桩底参数
                self.pmb[k] = pile_info.PMB
                self.peh[k] = pile_info.PEH
                self.pke[k] = pile_info.PKE
        
        # simu_pile - 设置模拟桩信息
        if self.snum > 0:
            # 读取KSCTR
            self.ksctr = np.array(Pile.simu_pile.simu_pile.KSCTR, dtype=int)
            raise NotImplementedError("Simulated piles are not supported yet.")
            
            # 为模拟桩单元刚度矩阵赋值
            is_val = self.pnum * 6  # 初始索引
            
            for k in range(self.snum):
                pile_type_id = self.ksctr[k]
                
                # 如果是负值模式（对角元素模式）
                if pile_type_id < 0 and abs(pile_type_id) in Pile.simu_pile.simu_pile.pile_types:
                    # 获取对角元素，这里假设是直接存储在pile_types中
                    diagonal_values = Pile.simu_pile.simu_pile.pile_types[abs(pile_type_id)]
                    
                    # 设置对角元素（每个自由度一个刚度值）
                    for ia in range(6):
                        is_val += 1
                        for ib in range(6):
                            self.esp[is_val, ib] = 0.0
                        # 设置对角元素
                        if hasattr(diagonal_values, f'K{ia+1}'):  # 假设对角元素命名为K1, K2, ..., K6
                            self.esp[is_val, ia] = getattr(diagonal_values, f'K{ia+1}')
                
                # 如果是正值模式（完整刚度矩阵模式）
                elif pile_type_id > 0 and pile_type_id in Pile.simu_pile.simu_pile.pile_types:
                    # 获取完整刚度矩阵
                    stiffness_matrix = Pile.simu_pile.simu_pile.pile_types[pile_type_id]
                    
                    # 设置完整刚度矩阵（6x6）
                    if hasattr(stiffness_matrix, 'matrix'):  # 假设完整矩阵存储在matrix属性中
                        matrix = stiffness_matrix.matrix
                        for ia in range(6):
                            is_val += 1
                            for ib in range(6):
                                self.esp[is_val, ib] = matrix[ia][ib]
        
        # 计算桩地上和地下段总长度
        self.zfr = np.zeros(self.pnum, dtype=float)
        self.zbl = np.zeros(self.pnum, dtype=float)
        
        for k in range(self.pnum):
            self.zfr[k] = np.sum(self.hfr[k, :int(self.nfr[k])])
            self.zbl[k] = np.sum(self.hbl[k, :int(self.nbl[k])])

        logger.opt(colors=True).info(f"Parameters initialized for <green>{Pile.name}</green>")

    def read_dat(self, file_path: Path = "*.dat") -> PileModel:
        """读取初始结构数据
        
        Args:
            file_path: 输入文件路径
            force: 外部荷载数组
            
        Returns:
            PileModel: 解析后的桩基础数据
        """
        
        with open(file_path, 'r') as self.input_file:
            input_text = self.input_file.read()
            self.Pile = PileModel(input_text=input_text)
            if not self.Pile.name:
                self.Pile.name = file_path.stem
        self.init_parameters(self.Pile)
        return self.Pile
        
    def btxy(self) -> tuple[np.ndarray, np.ndarray]:
        """计算桩的变形系数"""
        logger.debug("Calculating deformation factors of piles...")
        # 初始化变形系数
        self.btx = np.zeros((self.pnum, self.N_max_layer), dtype=float)
        self.bty = np.zeros((self.pnum, self.N_max_layer), dtype=float)
        
        # 计算桩在地面处的坐标
        gxy = np.zeros((self.pnum, 2), dtype=float)
        for k in range(self.pnum):
            gxy[k, 0] = self.pxy[k, 0] + self.zbl[k] * self.agl[k, 0]
            gxy[k, 1] = self.pxy[k, 1] + self.zbl[k] * self.agl[k, 1]
        
        # 计算桩间距
        for k in range(self.pnum):
            for k1 in range(k+1, self.pnum):
                s = np.sqrt((gxy[k, 0] - gxy[k1, 0])**2 + (gxy[k, 1] - gxy[k1, 1])**2) - (self.dob[k, 0] + self.dob[k1, 0]) / 2.0
                if s < 1.0:
                    # 桩间距小于1m，调用kinf1函数
                    kinf = np.zeros(2, dtype=float)
                    self.kinf1(0, self.pnum, self.dob, self.zbl, gxy, kinf, 0)
                    self.kinf1(1, self.pnum, self.dob, self.zbl, gxy, kinf, 1)
                    break
        else:
            # 桩间距大于1m，调用kinf2函数
            kinf = np.zeros(2, dtype=float)
            self.kinf2(0, self.pnum, self.dob, self.zbl, gxy, kinf, 0)
            self.kinf2(1, self.pnum, self.dob, self.zbl, gxy, kinf, 1)
        
        # 计算每个桩的变形系数
        for k in range(self.pnum):
            if k > 0:
                # 检查是否有相同控制信息的桩，如果有则复制变形系数
                for k1 in range(k):
                    if self.kctr[k] == self.kctr[k1]:
                        for ia in range(int(self.nbl[k1])):
                            self.btx[k, ia] = self.btx[k1, ia]
                            self.bty[k, ia] = self.bty[k1, ia]
                        break
                else:
                    # 计算新桩的变形系数
                    ka = 1.0
                    if self.ksh[k] == 1:
                        ka = 0.9
                    
                    for ia in range(int(self.nbl[k])):
                        bx1 = ka * kinf[0] * (self.dob[k, ia] + 1.0)
                        by1 = ka * kinf[1] * (self.dob[k, ia] + 1.0)
                        a, b = self.eaj(self.ksh[k], self.pke[k], self.dob[k, ia])
                        self.btx[k, ia] = (self.pmt[k, ia] * bx1 / (self.peh[k] * b))**0.2
                        self.bty[k, ia] = (self.pmt[k, ia] * by1 / (self.peh[k] * b))**0.2
            else:
                # 计算第一个桩的变形系数
                ka = 1.0
                if self.ksh[k] == 1:
                    ka = 0.9
                
                for ia in range(int(self.nbl[k])):
                    bx1 = ka * kinf[0] * (self.dob[k, ia] + 1.0)
                    by1 = ka * kinf[1] * (self.dob[k, ia] + 1.0)
                    a, b = self.eaj(self.ksh[k], self.pke[k], self.dob[k, ia])
                    self.btx[k, ia] = (self.pmt[k, ia] * bx1 / (self.peh[k] * b))**0.2
                    self.bty[k, ia] = (self.pmt[k, ia] * by1 / (self.peh[k] * b))**0.2
        logger.success("Deformation factors of piles calculated!")
        return self.btx, self.bty

    def kinf1(self, im, pnum, dob, zbl, gxy, kinf, idx):
        """计算影响系数 - 桩间距小于1m的情况"""
        aa = []
        dd = []
        zz = []
        
        # 收集坐标方向im上不同的桩
        aa.append(gxy[0, im])
        dd.append(dob[0, 0])
        zz.append(zbl[0])
        
        for k in range(1, pnum):
            if gxy[k, im] not in aa:
                aa.append(gxy[k, im])
                dd.append(dob[k, 0])
                zz.append(zbl[k])
        
        # 计算影响系数
        kinf_temp = np.zeros(1, dtype=float)
        self.kinf3(len(aa), aa, dd, zz, kinf_temp, 0)
        kinf[idx] = kinf_temp[0]

    def kinf2(self, im, pnum, dob, zbl, gxy, kinf, idx):
        """计算影响系数 - 桩间距大于1m的情况"""
        im1 = 1 if im == 0 else 0
        nrow = 0
        nin = {}
        in_arr = []
        nok = []
        
        # 按im1方向分组
        for k in range(pnum):
            found = False
            for k1 in range(k):
                if gxy[k, im1] == gxy[k1, im1]:
                    na = nin[k1]
                    if na < len(in_arr):
                        in_arr[na] += 1
                        nok[na].append(k)
                    found = True
                    break
            
            if not found:
                nin[k] = nrow
                in_arr.append(1)
                nok.append([k])
                nrow += 1
        
        # 查找最小影响系数
        kmin = 1.0
        for i in range(nrow):
            aa = []
            dd = []
            zz = []
            
            for j in range(in_arr[i]):
                k = nok[i][j]
                aa.append(gxy[k, im])
                dd.append(dob[k, 0])
                zz.append(zbl[k])
            
            kinf_temp = np.zeros(1, dtype=float)
            self.kinf3(len(aa), aa, dd, zz, kinf_temp, 0)
            
            if kinf_temp[0] < kmin:
                kmin = kinf_temp[0]
        
        kinf[idx] = kmin

    def kinf3(self, in_val, aa, dd, zz, kinf, idx):
        """计算桩行的影响系数"""
        if in_val == 1:
            kinf[idx] = 1.0
            return
        
        # 计算影响范围
        ho = []
        for i in range(in_val):
            ho_val = 3.0 * (dd[i] + 1.0)
            if ho_val > zz[i]:
                ho_val = zz[i]
            ho.append(ho_val)
        
        # 查找最小间距
        lo = 100.0
        hoo = 0.0
        
        for i in range(in_val):
            for i1 in range(i+1, in_val):
                s = abs(aa[i] - aa[i1]) - (dd[i] + dd[i1]) / 2.0
                if s < lo:
                    lo = s
                    hoo = max(ho[i], ho[i1])
        
        # 计算影响系数
        if lo >= 0.6 * hoo:
            kinf[idx] = 1.0
        else:
            c = self.parc(in_val)
            kinf[idx] = c + (1.0 - c) * lo / (0.6 * hoo)

    def parc(self, in_val):
        """计算桩群系数"""
        if in_val == 1:
            return 1.0
        elif in_val == 2:
            return 0.6
        elif in_val == 3:
            return 0.5
        else:  # in_val >= 4
            return 0.45

    def area(self):
        """计算桩底面积"""

        # 初始化桩底面积数组
        self.ao = np.zeros(self.pnum, dtype=float)
        
        # 计算桩底坐标
        bxy = np.zeros((self.pnum, 2), dtype=float)
        w = np.zeros(self.pnum, dtype=float)
        smin = np.ones(self.pnum, dtype=float) * 100.0
        
        for k in range(self.pnum):
            bxy[k, 0] = self.pxy[k, 0] + (self.zfr[k] + self.zbl[k]) * self.agl[k, 0]
            bxy[k, 1] = self.pxy[k, 1] + (self.zfr[k] + self.zbl[k]) * self.agl[k, 1]
            
            if self.ksu[k] > 2:
                if self.nbl[k] != 0:
                    w[k] = self.dob[k, int(self.nbl[k]-1)]
                else:
                    w[k] = self.dof[k, int(self.nfr[k]-1)]
                continue
            
            # 计算桩底宽度
            w[k] = 0.0
            ag = math.atan(math.sqrt(1 - self.agl[k, 2]**2) / self.agl[k, 2])
            
            for ia in range(int(self.nbl[k])):
                w[k] += self.hbl[k, ia] * (math.sin(ag) - self.agl[k, 2] * 
                                         math.tan(ag - self.pfi[k, ia] * math.pi / 720.0))
            
            w[k] = w[k] * 2 + self.dob[k, 0]
        
        # 计算桩间最小距离
        for k in range(self.pnum):
            for ia in range(k+1, self.pnum):
                s = math.sqrt((bxy[k, 0] - bxy[ia, 0])**2 + (bxy[k, 1] - bxy[ia, 1])**2)
                if s < smin[k]:
                    smin[k] = s
                if s < smin[ia]:
                    smin[ia] = s
        
        # 确定使用最小宽度并计算桩底面积
        for k in range(self.pnum):
            if smin[k] < w[k]:
                w[k] = smin[k]
            
            if self.ksh[k] == 0:  # 圆形
                self.ao[k] = math.pi * w[k]**2 / 4.0
            else:  # 方形
                self.ao[k] = w[k]**2

        logger.debug("Pile bottom areas calculated!")
        return self.ao

    def stn(self, k, zbl_k, ao_k, rzz):
        """计算单桩轴向刚度"""
        # 确定桩底约束条件系数
        if self.ksu[k] == 1:
            pkc = 0.5
        elif self.ksu[k] == 2:
            pkc = 0.667
        else:  # self.ksu[k] > 2
            pkc = 1.0
        
        # 计算轴向挠度
        x = 0.0
        
        # 地上段挠度
        for ia in range(int(self.nfr[k])):
            a, b = self.eaj(self.ksh[k], self.pke[k], self.dof[k, ia])
            x += self.hfr[k, ia] / (self.peh[k] * a)
        
        # 地下段挠度
        for ia in range(int(self.nbl[k])):
            a, b = self.eaj(self.ksh[k], self.pke[k], self.dob[k, ia])
            x += pkc * self.hbl[k, ia] / (self.peh[k] * a)
        
        # 桩端挠度
        if self.ksu[k] <= 2:
            x += 1.0 / (self.pmb[k] * zbl_k * ao_k)
        else:  # self.ksu[k] > 2
            x += 1.0 / (self.pmb[k] * ao_k)
        
        # 刚度为挠度的倒数
        rzz[0] = 1.0 / x

    def eaj(self, j, pke, d_o):
        """计算桩截面性质"""
        if j == 0:  # 圆形
            a = math.pi * d_o**2 / 4.0
            b = pke * math.pi * d_o**4 / 64.0
        else:  # 方形
            a = d_o**2
            b = pke * d_o**4 / 12.0
        
        return a, b

    def stiff_n(self):
        """计算每个桩的轴向刚度"""
        if not hasattr(self,'ao'):
            self.area()
        logger.debug("Calculating axial stiffness of piles...")
        # 初始化轴向刚度数组
        self.rzz = np.zeros(self.pnum, dtype=float)
        
        # 计算第一个桩的轴向刚度
        self.stn(0, self.zbl[0], self.ao[0], self.rzz[0:1])
        
        # 计算其他桩的轴向刚度
        for k in range(1, self.pnum):
            # 检查是否有相同控制信息和底面积的桩
            for ia in range(k):
                if self.kctr[k] == self.kctr[ia] and self.ao[k] == self.ao[ia]:
                    self.rzz[k] = self.rzz[ia]
                    break
            else:
                # 计算新桩的轴向刚度
                self.stn(k, self.zbl[k], self.ao[k], self.rzz[k:k+1])
        logger.success("Axial stiffness of piles calculated!")
        return self.rzz

    def rltfr(self, nfr, ej, hfr, kfr):
        """计算桩地上段的关系矩阵"""
        # 计算第一段的关系矩阵
        self.mfree(ej[0], hfr[0], kfr)
        
        # 逐段组合关系矩阵
        for ia in range(1, nfr):
            r = np.zeros((4, 4), dtype=float)
            self.mfree(ej[ia], hfr[ia], r)
            
            # 矩阵相乘
            rm = np.dot(kfr, r)
            kfr[:] = rm

    def mfree(self, ej, h, r):
        """计算一个桩段的关系矩阵"""
        # 初始化为单位矩阵
        r[:] = np.eye(4)
        
        # 填充关系矩阵元素
        r[0, 1] = h
        r[0, 2] = h**3 / (6.0 * ej)
        r[0, 3] = -h**2 / (2.0 * ej)
        r[1, 2] = h**2 / (2.0 * ej)
        r[1, 3] = -h / ej
        r[3, 2] = -h

    def combx(self, kbx, kfr, kx):
        """组合桩地上段和地下段关系矩阵"""
        # 修改地下段矩阵
        kbx_copy = kbx.copy()
        kbx_copy[:, 3] = -kbx_copy[:, 3]
        
        # 矩阵相乘
        kx[:] = np.dot(kbx_copy, kfr)

    def cndtn(self, ksu, kx, ky, rzz, ke):
        """计算考虑边界条件的桩单元刚度"""
        # 初始化刚度矩阵
        ke.fill(0.0)
        
        # 处理X方向刚度
        at = np.zeros((2, 2), dtype=float)
        self.dvsn(ksu, kx, at)
        
        ke[0, 0] = at[0, 0]
        ke[0, 4] = at[0, 1]
        ke[4, 0] = at[1, 0]
        ke[4, 4] = at[1, 1]
        
        # 处理Y方向刚度
        self.dvsn(ksu, ky, at)
        
        ke[1, 1] = at[0, 0]
        ke[1, 3] = -at[0, 1]
        ke[3, 1] = -at[1, 0]
        ke[3, 3] = at[1, 1]
        
        # 轴向和扭转刚度
        ke[2, 2] = rzz
        ke[5, 5] = 0.1 * (ke[3, 3] + ke[4, 4])

    def dvsn(self, ksu, kxy, at):
        """处理桩的边界条件"""
        # 分解矩阵
        a11 = kxy[0:2, 0:2]
        a12 = kxy[0:2, 2:4]
        a21 = kxy[2:4, 0:2]
        a22 = kxy[2:4, 2:4]
        
        if ksu == 4:  # 固定端约束
            av = np.linalg.inv(a12)
            at[:] = -np.dot(av, a11)
        else:  # 自由端约束
            av = np.linalg.inv(a22)
            at[:] = -np.dot(av, a21)

    def trnsfr(self, x, y, z, tk):
        """形成桩的转换矩阵"""
        # 计算单位方向向量参数
        b = math.sqrt(y**2 + z**2)
        
        # 初始化转换矩阵
        tk.fill(0.0)
        
        # 填充转换矩阵
        tk[0, 0] = b
        tk[0, 2] = x
        tk[1, 0] = -x * y / b
        tk[1, 1] = z / b
        tk[1, 2] = y
        tk[2, 0] = -x * z / b
        tk[2, 1] = -y / b
        tk[2, 2] = z
        
        # 右下角为左上角的复制
        for i in range(3):
            for j in range(3):
                tk[i+3, j+3] = tk[i, j]

    def pstiff(self):
        """计算桩单元刚度"""
        if not hasattr(self,'rzz'):
            self.rzz = self.stiff_n()
        
        if not hasattr(self,'btx') or not hasattr(self,'bty'):
            self.btxy()

        logger.debug("Calculating lateral stiffness of piles...")
        # 桩单元刚度
        self.esp = np.zeros((self.N_max_pile**2, 6), dtype=float)
        
        for k in range(self.pnum):
            # 如果桩无地下段，使用单位矩阵
            if self.nbl[k] == 0:
                kbx = np.eye(4)
                kby = np.eye(4)
            else:
                # 收集桩段信息
                h = np.zeros(self.N_max_calc_points, dtype=float)
                bt1 = np.zeros(self.N_max_layer, dtype=float)
                bt2 = np.zeros(self.N_max_layer, dtype=float)
                ej = np.zeros(self.N_max_layer, dtype=float)
                
                h[0] = 0.0
                for ia in range(int(self.nbl[k])):
                    bt1[ia] = self.btx[k, ia]
                    bt2[ia] = self.bty[k, ia]
                    a, b = self.eaj(self.ksh[k], self.pke[k], self.dob[k, ia])
                    ej[ia] = self.peh[k] * b
                    h[ia+1] = h[ia] + self.hbl[k, ia]
                
                # 计算地下段关系矩阵
                kbx = np.zeros((4, 4), dtype=float)
                kby = np.zeros((4, 4), dtype=float)
                self.rltmtx(int(self.nbl[k]), bt1, bt2, ej, h, kbx, kby)
            
            # 如果桩无地上段，直接处理
            if self.nfr[k] == 0:
                kx = kbx.copy()
                ky = kby.copy()
                kx[:, 3] = -kx[:, 3]
                ky[:, 3] = -ky[:, 3]
            else:
                # 计算地上段关系矩阵
                h = np.zeros(self.N_max_calc_points, dtype=float)
                ej = np.zeros(self.N_max_layer, dtype=float)
                
                for ia in range(int(self.nfr[k])):
                    a, b = self.eaj(self.ksh[k], self.pke[k], self.dof[k, ia])
                    ej[ia] = self.peh[k] * b
                    h[ia] = self.hfr[k, ia]
                
                kfr = np.zeros((4, 4), dtype=float)
                self.rltfr(int(self.nfr[k]), ej, h, kfr)
                
                # 组合地上段和地下段关系矩阵
                kx = np.zeros((4, 4), dtype=float)
                ky = np.zeros((4, 4), dtype=float)
                self.combx(kbx, kfr, kx)
                self.combx(kby, kfr, ky)
            
            # 计算考虑边界条件的桩单元刚度
            ke = np.zeros((6, 6), dtype=float)
            self.cndtn(self.ksu[k], kx, ky, self.rzz[k], ke)
            
            # 保存桩的单元刚度
            for i in range(6):
                for j in range(6):
                    self.esp[(k-1)*6+i, j] = ke[i, j]
        
        logger.success("Lateral stiffness of piles calculated!")
        return self.esp

    def rltmtx(self, nbl, bt1, bt2, ej, h, kbx, kby):
        """计算桩地下段的关系矩阵"""
        # 计算第一段的关系矩阵
        self.saa(bt1[0], ej[0], h[0], h[1], kbx)
        
        # 逐段组合关系矩阵
        for ia in range(1, nbl):
            a1 = kbx.copy()
            a2 = np.zeros((4, 4), dtype=float)
            self.saa(bt1[ia], ej[ia], h[ia], h[ia+1], a2)
            kbx[:] = np.dot(a2, a1)
        
        # 检查X和Y方向的变形系数是否相同
        is_same = True
        for ia in range(nbl):
            if abs(bt2[ia] - bt1[ia]) > 1.0e-10:
                is_same = False
                break
        
        if is_same:
            # 如果相同，直接复制X方向的关系矩阵
            kby[:] = kbx
        else:
            # 如果不同，重新计算Y方向的关系矩阵
            self.saa(bt2[0], ej[0], h[0], h[1], kby)
            
            for ia in range(1, nbl):
                a1 = kby.copy()
                a2 = np.zeros((4, 4), dtype=float)
                self.saa(bt2[ia], ej[ia], h[ia], h[ia+1], a2)
                kby[:] = np.dot(a2, a1)

    def saa(self, bt, ej, h1, h2, ai):
        """计算一个非自由桩段的关系矩阵"""
        # 计算两个高度处的系数矩阵
        ai1 = np.zeros((4, 4), dtype=float)
        ai2 = np.zeros((4, 4), dtype=float)
        
        self.param(bt, ej, h1, ai1)
        self.param(bt, ej, h2, ai2)
        
        # 计算关系矩阵
        ai3 = np.linalg.inv(ai1)
        ai[:] = np.dot(ai2, ai3)
        
        # 调整单位系统
        for i in range(2):
            for j in range(2):
                ai[i, j+2] = ai[i, j+2] / ej
                ai[i+2, j] = ai[i+2, j] * ej
        
        # 调整矩阵顺序
        ai[[2, 3]] = ai[[3, 2]]
        ai[:, [2, 3]] = ai[:, [3, 2]]

    def param(self, bt, ej, x, aa):
        """计算系数矩阵"""
        # 计算参数
        y = bt * x
        if y > 6.0:
            y = 6.0
        
        # 计算幂级数项的值
        a1, b1, c1, d1, a2, b2, c2, d2 = self.param1(y)
        a3, b3, c3, d3, a4, b4, c4, d4 = self.param2(y)
        
        # 填充系数矩阵
        aa[0, 0] = a1
        aa[0, 1] = b1 / bt
        aa[0, 2] = 2 * c1 / bt**2
        aa[0, 3] = 6 * d1 / bt**3
        aa[1, 0] = a2 * bt
        aa[1, 1] = b2
        aa[1, 2] = 2 * c2 / bt
        aa[1, 3] = 6 * d2 / bt**2
        aa[2, 0] = a3 * bt**2
        aa[2, 1] = b3 * bt
        aa[2, 2] = 2 * c3
        aa[2, 3] = 6 * d3 / bt
        aa[3, 0] = a4 * bt**3
        aa[3, 1] = b4 * bt**2
        aa[3, 2] = 2 * c4 * bt
        aa[3, 3] = 6 * d4

    def param1(self, y):
        """近似计算幂级数项的值 - 第一组"""
        a1 = 1 - y**5/120.0 + y**10/6.048e5 - y**15/1.9813e10 + y**20/2.3038e15 - y**25/6.9945e20
        b1 = y - y**6/360.0 + y**11/2851200 - y**16/1.245e11 + y**21/1.7889e16 - y**26/6.4185e21
        c1 = y**2/2.0 - y**7/1680 + y**12/1.9958e7 - y**17/1.14e12 + y**22/2.0e17 - y**27/8.43e22
        d1 = y**3/6.0 - y**8/10080 + y**13/1.7297e8 - y**18/1.2703e13 + y**23/2.6997e18 - y**28/1.33e24
        a2 = -y**4/24.0 + y**9/6.048e4 - y**14/1.3209e9 + y**19/1.1519e14 - y**24/2.7978e19
        b2 = 1 - y**5/60.0 + y**10/2.592e5 - y**15/7.7837e9 + y**20/8.5185e14 - y**25/2.4686e20
        c2 = y - y**6/240.0 + y**11/1.6632e6 - y**16/6.7059e10 + y**21/9.0973e15 - y**26/3.1222e21
        d2 = y**2/2 - y**7/1260 + y**12/1.3305e7 - y**17/7.0572e11 + y**22/1.1738e17 - y**27/4.738e22
        
        return a1, b1, c1, d1, a2, b2, c2, d2

    def param2(self, y):
        """近似计算幂级数项的值 - 第二组"""
        a3 = -y**3/6.0 + y**8/6.72e3 - y**13/9.435e7 + y**18/6.0626e12 - y**23/1.1657e18
        b3 = -y**4/12.0 + y**9/25920 - y**14/5.1892e8 + y**19/4.2593e13 - y**24/9.8746e18
        c3 = 1 - y**5/40.0 + y**10/151200 - y**15/4.1912e9 + y**20/4.332e14 - y**25/1.2009e20
        d3 = y - y**6/180.0 + y**11/1108800 - y**16/4.1513e10 + y**21/5.3354e15 - y**26/1.7543e21
        a4 = -y**2/2.0 + y**7/840.0 - y**12/7.257e6 + y**17/3.3681e11 - y**22/5.0683e16
        b4 = -y**3/3.0 + y**8/2880 - y**13/3.7066e7 + y**18/2.2477e12 - y**23/4.1144e17
        c4 = -y**4/8 + y**9/1.512e4 - y**14/2.7941e8 + y**19/2.166e13 - y**24/4.8034e18
        d4 = 1 - y**5/30 + y**10/100800 - y**15/2.5946e9 + y**20/2.5406e14 - y**25/6.7491e19
        
        return a3, b3, c3, d3, a4, b4, c4, d4

    def tmatx(self, x, y, tu=None):
        """计算单元坐标系的转换矩阵"""
        if tu is None:
            tu = np.zeros((6, 6), dtype=float)
        
        # 初始化为单位矩阵
        np.fill_diagonal(tu, 1.0)
        
        # 填充转换矩阵的非对角元素
        tu[0, 5] = -y
        tu[1, 5] = x
        tu[2, 3] = y
        tu[2, 4] = -x
        
        return tu

    @property
    def K(self):
        """计算桩基础帽的刚度"""
        if not hasattr(self, 'esp'):
            self.pstiff()

        K = np.zeros((6, 6), dtype=float)
        for k in range(self.pnum + self.snum):
            # 获取桩的单元刚度
            a = np.zeros((6, 6), dtype=float)
            for i in range(6):
                for j in range(6):
                    a[i, j] = self.esp[(k-1)*6+i, j]
            
            # 应用转换矩阵
            if k < self.pnum:
                # 非模拟桩需要考虑倾斜方向
                tk = np.zeros((6, 6), dtype=float)
                self.trnsfr(self.agl[k, 0], self.agl[k, 1], self.agl[k, 2], tk)
                tk1 = tk.T
                a1 = np.dot(tk, a)
                a = np.dot(a1, tk1)
                
                x = self.pxy[k, 0]
                y = self.pxy[k, 1]
            else:
                # 模拟桩
                x = self.sxy[k-self.pnum, 0]
                y = self.sxy[k-self.pnum, 1]
            
            # 应用位置转换矩阵
            tu = np.zeros((6, 6), dtype=float)
            self.tmatx(x, y, tu)
            tn = tu.T
            b = np.dot(a, tu)
            a = np.dot(tn, b)
            
            # 累加到整体刚度矩阵
            K += a
        return K
    
    def K_pile(self, ino:int = None):
        """计算桩基础指定桩的刚度"""
        if ino is None:
            if hasattr(self, 'ino'):
                ino = self.ino
            else:
                raise ValueError("请指定桩号或修改输入文件！")
        
        # 计算指定桩的刚度
        K_pile = np.zeros((6, 6), dtype=float)
        for i in range(6):
            for j in range(6):
                K_pile[i, j] = self.esp[(ino-1)*6+i, j]
        return K_pile
    
    def disp_cap(self, force:np.ndarray = None, print_in_cli:bool = False):
        """计算桩基础帽的位移"""
        # 求解位移
        if force is None:
            if hasattr(self, 'force'):
                force = self.force
            else:
                raise ValueError("请提供力向量！")
        
        # 计算整个桩基础的刚度
        K = self.K
        
        # 求解位移
        disp = np.linalg.solve(K, force)
        if print_in_cli:
            # 输出位移结果
            print("\n       *****************************************************************************************************\n")
            print("               DISPLACEMENTS AT THE CAP CENTER OF PILE FOUNDATION\n")
            print("       *****************************************************************************************************\n")
            print(f"\n                Movement in the direction of X axis : UX= {disp[0]:12.4e} (m)\n")
            print(f"                Movement in the direction of Y axis : UY= {disp[1]:12.4e} (m)\n")
            print(f"                Movement in the direction of Z axis : UZ= {disp[2]:12.4e} (m)\n")
            print(f"                Rotational angle  around X axis :     SX= {disp[3]:12.4e} (rad)\n")
            print(f"                Rotational angle  around Y axis :     SY= {disp[4]:12.4e} (rad)\n")
            print(f"                Rotational angle  around Z axis :     SZ= {disp[5]:12.4e} (rad)\n")
        return disp
    
    def disp_piles(self, force:np.ndarray = None, print_in_cli:bool = False):
        """计算桩基础各桩的位移"""
        # 求解位移
        if force is None:
            if hasattr(self, 'force'):
                force = self.force
            else:
                raise ValueError("请提供力向量！")
        
        disp = self.disp_cap(force)
        
        self.duk = np.zeros((self.pnum, 6), dtype=float)
        for k in range(self.pnum):
            # 应用位置转换矩阵
            tu = np.zeros((6, 6), dtype=float)
            self.tmatx(self.pxy[k, 0], self.pxy[k, 1], tu)
            c1 = np.dot(tu, disp)
            
            # 应用方向转换矩阵
            tk = np.zeros((6, 6), dtype=float)
            self.trnsfr(self.agl[k, 0], self.agl[k, 1], self.agl[k, 2], tk)
            tk1 = tk.T
            c = np.dot(tk1, c1)
            
            # 保存桩的局部位移
            for i in range(6):
                self.duk[k, i] = c[i]
                
        if print_in_cli:
            # 输出位移结果
            print("\n       *****************************************************************************************************\n")
            print("               DISPLACEMENTS AT EACH PILES\n")
            print("       *****************************************************************************************************\n")
            for k in range(self.pnum):
                print(f"Pile {k+1}: UX= {duk[k, 0]:12.4e} (m), UY= {duk[k, 1]:12.4e} (m), UZ= {duk[k, 2]:12.4e} (m), SX= {duk[k, 3]:12.4e} (rad), SY= {duk[k, 4]:12.4e} (rad), SZ= {duk[k, 5]:12.4e} (rad)")
            
        return self.duk
    
    def eforce(self, force:np.ndarray = None, print_in_cli:bool = False):
        """计算桩体的位移和内力"""
        if not hasattr(self, 'duk'):
            self.disp_piles(force)
        
        # 计算每个桩的位移和内力
        for k in range(self.pnum):
            # 获取桩的单元刚度和位移
            ce = self.duk[k, :]
            se = np.zeros((6, 6), dtype=float)
            for i in range(6):
                for j in range(6):
                    se[i, j] = self.esp[(k-1)*6+i, j]
            
            # 计算桩顶部的内力
            pe = np.dot(se, ce)
            
            # 初始化数组
            zh = np.zeros(self.N_max_calc_points, dtype=float)
            fx = np.zeros((self.N_max_calc_points, 4), dtype=float)
            fy = np.zeros((self.N_max_calc_points, 4), dtype=float)
            fz = np.zeros(self.N_max_calc_points, dtype=float)
            psx = np.zeros(self.N_max_calc_points, dtype=float)
            psy = np.zeros(self.N_max_calc_points, dtype=float)
            
            # 设置第一个节点的信息
            zh[0] = 0.0
            fx[0, 0] = ce[0]
            fx[0, 1] = ce[4]
            fx[0, 2] = pe[0]
            fx[0, 3] = pe[4]
            fy[0, 0] = ce[1]
            fy[0, 1] = ce[3]
            fy[0, 2] = pe[1]
            fy[0, 3] = pe[3]
            fz[0] = pe[2]
            
            # 沿桩长度计算位移和内力
            nsum = 0
            
            # 地上段计算
            for ia in range(int(self.nfr[k])):
                hl = self.hfr[k, ia] / self.nsf[k, ia]
                a, b = self.eaj(self.ksh[k], self.pke[k], self.dof[k, ia])
                ej = self.peh[k] * b
                r = np.zeros((4, 4), dtype=float)
                self.mfree(ej, hl, r)
                
                for in_val in range(int(self.nsf[k, ia])):
                    xa = fx[nsum, :]
                    xc = fy[nsum, :]
                    xc[1] = -xc[1]
                    xc[3] = -xc[3]
                    
                    xb = np.dot(r, xa)
                    xd = np.dot(r, xc)
                    
                    nsum += 1
                    fx[nsum, :] = xb
                    fy[nsum, :] = xd
                    fy[nsum, 1] = -xd[1]
                    fy[nsum, 3] = -xd[3]
                    zh[nsum] = zh[nsum-1] + hl
                    fz[nsum] = fz[nsum-1]
            
            # 保存地面位置索引
            ig = nsum
            zg = zh[nsum]
            psx[nsum] = 0.0
            psy[nsum] = 0.0
            
            # 地下段计算
            for ia in range(int(self.nbl[k])):
                hl = self.hbl[k, ia] / self.nsg[k, ia]
                a, b = self.eaj(self.ksh[k], self.pke[k], self.dob[k, ia])
                ej = self.peh[k] * b
                
                for in_val in range(int(self.nsg[k, ia])):
                    h1 = zh[nsum] - zg
                    h2 = h1 + hl
                    
                    xa = fx[nsum, :].copy()
                    xc = fy[nsum, :].copy()
                    xa[3] = -xa[3]
                    xc[1] = -xc[1]
                    
                    r = np.zeros((4, 4), dtype=float)
                    self.saa(self.btx[k, ia], ej, h1, h2, r)
                    xb = np.dot(r, xa)
                    
                    if abs(self.btx[k, ia] - self.bty[k, ia]) > 1.0e-3:
                        r = np.zeros((4, 4), dtype=float)
                        self.saa(self.bty[k, ia], ej, h1, h2, r)
                    
                    xd = np.dot(r, xc)
                    
                    nsum += 1
                    fx[nsum, :] = xb
                    fy[nsum, :] = xd
                    fx[nsum, 3] = -xb[3]
                    fy[nsum, 1] = -xd[1]
                    zh[nsum] = zh[nsum-1] + hl
                    psx[nsum] = fx[nsum, 0] * h2 * self.pmt[k, ia]
                    psy[nsum] = fy[nsum, 0] * h2 * self.pmt[k, ia]
                    
                    if self.ksu[k] >= 3:
                        fz[nsum] = fz[nsum-1]
                    else:
                        fz[nsum] = fz[ig] * (1.0 - h2**2 / self.zbl[k]**2)
            
            # 输出桩体位移和内力到输出文件
            print("       ****************************************************************************************\n")
            print(f"                                  NO. {k} # PILE\n")
            print("       ****************************************************************************************\n")
            print(f"\n            Coordinator of the pile: (x,y) = ({self.pxy[k, 0]:12.4e} ,{self.pxy[k, 1]:12.4e} )\n\n")
            print("            Displacements and internal forces at the top of pile:\n")
            print(f"\n               UX= {ce[0]:12.4e} (m)         NX= {pe[0]:12.4e} (t)\n")
            print(f"               UY= {ce[1]:12.4e} (m)         NY= {pe[1]:12.4e} (t)\n")
            print(f"               UZ= {ce[2]:12.4e} (m)         NZ= {pe[2]:12.4e} (t)\n")
            print(f"               SX= {ce[3]:12.4e} (rad)       MX= {pe[3]:12.4e} (t*m)\n")
            print(f"               SY= {ce[4]:12.4e} (rad)       MY= {pe[4]:12.4e} (t*m)\n")
            print(f"               SZ= {ce[5]:12.4e} (rad)       MZ= {pe[5]:12.4e} (t*m)\n\n")
            print
            print("       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            print("                                Displacements of the pile body and\n")
            print("                             Compression stresses of soil (PSX,PSY)\n")
            print("       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            print("\n               Z            UX            UY            SX            SY            PSX            PSY\n")
            print("              (m)           (m)           (m)          (rad)         (rad)         (t/m2)         (t/m2)\n\n")
            
            # 输出地上段数据
            for i in range(ig):
                line = f"       {zh[i]:14.4e}{fx[i, 0]:14.4e}{fy[i, 0]:14.4e}{fy[i, 1]:14.4e}{fx[i, 1]:14.4e}"
                print(line + "\n")
            
            # 输出地下段数据
            for i in range(ig, nsum+1):
                line = f"       {zh[i]:14.4e}{fx[i, 0]:14.4e}{fy[i, 0]:14.4e}{fy[i, 1]:14.4e}{fx[i, 1]:14.4e}{psx[i]:14.4e}{psy[i]:14.4e}"
                print(line + "\n")
            
            print("\n\n")
            print("       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            print("                                Internal forces of the pile body\n")
            print("       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            print("\n                  Z              NX              NY              NZ              MX              MY\n")
            print("                 (m)             (t)             (t)             (t)            (t*m)           (t*m)\n\n")
            
            # 输出内力数据
            for i in range(nsum+1):
                line = f"       {zh[i]:16.4e}{fx[i, 2]:16.4e}{fy[i, 2]:16.4e}{fz[i]:16.4e}{fy[i, 3]:16.4e}{fx[i, 3]:16.4e}"
                print(line + "\n")
            
            # 输出位移和内力数据到pos文件
            # 为地上段设置零土压力
            for i in range(ig):
                psx[i] = 0.0
                psy[i] = 0.0
            
            # 写入桩号和节点数
            print(f"{k} {nsum+1}\n")
            # 写入桩顶坐标
            print(f"{self.pxy[k, 0]:14.4e} {self.pxy[k, 1]:14.4e}\n")
            # 写入各节点Z坐标
            line = " ".join([f"{zh[i]:14.4e}" for i in range(nsum+1)])
            print(line + "\n")
            # 写入X方向变形和内力
            for i in range(nsum+1):
                line = " ".join([f"{fx[i, j]:14.4e}" for j in range(4)])
                self.pos_file.write(line + "\n")
            # 写入Y方向变形和内力
            for i in range(nsum+1):
                line = " ".join([f"{fy[i, j]:14.4e}" for j in range(4)])
                print(line + "\n")
            # 写入Z方向内力
            line = " ".join([f"{fz[i]:14.4e}" for i in range(nsum+1)])
            print(line + "\n")
            # 写入X方向土压力
            line = " ".join([f"{psx[i]:14.4e}" for i in range(nsum+1)])
            print(line + "\n")
            # 写入Y方向土压力
            line = " ".join([f"{psy[i]:14.4e}" for i in range(nsum+1)])
            print(line + "\n")
    
    def cli(self):
        """运行程序的主函数"""
        # 显示程序头
        print(self.welcome_message)
        
        # 读取数据文件名
        print("Please enter data filename:")
        fname = input().strip()
        
        # 构建输入输出文件名
        input_filename = self.f_name(fname, '.dat')
        output_filename = self.f_name(fname, '.out')
        pos_filename = self.f_name(fname, '.pos')
        
        # 打开文件
        self.input_file = open(input_filename, 'r')
        self.output_file = open(output_filename, 'w')
        self.pos_file = open(pos_filename, 'w')
        
        # 输出程序头到输出文件
        self.head2()
        
        # 初始化数据
        print("       *** To read input information ***\n")
        self.read_dat()
        
        # 执行计算流程 (使用拆分到calculation文件夹的计算模块)
        so = run_calculation(self, jctr, ino, pnum, snum, force, zfr, zbl)
        
        # 关闭文件
        self.input_file.close()
        self.output_file.close()
        self.pos_file.close()
        
        print("\n程序运行完成，结果已保存到 %s 和 %s 文件中。" % (output_filename, pos_filename))

if __name__ == "__main__":
    pile = PileManager(debug = True)
    pile.read_dat(Path("tests/Test-1-2.dat"))
    
    np.set_printoptions(linewidth=200, precision=2, suppress=True)
    # print(f"Pile stiffness matrix K:\n{pile.K}")
    

    np.testing.assert_allclose(pile.K, [
        [ 3.75361337e+06,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.65098740e+07,  0.00000000e+00],
        [ 0.00000000e+00,  3.68517766e+06,  0.00000000e+00, -1.63086648e+07, 0.00000000e+00,  6.98491931e-10],
        [ 0.00000000e+00,  0.00000000e+00,  3.40590554e+07,  0.00000000e+00, 1.86264515e-09,  0.00000000e+00],
        [ 0.00000000e+00, -1.62731362e+07,  0.00000000e+00,  4.64474149e+09, 0.00000000e+00, -2.79396772e-09],
        [ 1.64737208e+07,  0.00000000e+00,  1.86264515e-09,  0.00000000e+00, 5.76996816e+08,  0.00000000e+00],
        [ 0.00000000e+00,  6.98491931e-10,  0.00000000e+00, -9.31322575e-10, 0.00000000e+00,  5.72172846e+08]
    ], rtol=1e-5, atol=1e-8)
    
    # ino = 5
    # print(f"Pile {ino} stiffness matrix:\n{pile.K_pile(ino)}")
    
    # np.set_printoptions(linewidth=200, precision=4, suppress=True)
    force = np.array([22927.01, 0, 40702.94, 0, -332015.23, 0])

    print(f"Cap displacement:\n{pile.disp_cap(force)}")
    # print(f"Pile displacement:\n{pile.disp_piles(force)}")
    
    print(f"Pile forces:\n{pile.eforce(force)}")