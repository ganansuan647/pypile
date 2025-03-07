#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BCAD_PILE Python版本
这个程序旨在执行桥梁桩基础的空间静力分析
"""

import numpy as np
import math
import os

class BCADPile:
    def __init__(self):
        # 非模拟桩信息
        self.pxy = np.zeros((1000, 2), dtype=float)  # 桩的坐标
        self.kctr = np.zeros(1000, dtype=int)        # 桩的控制信息
        self.ksh = np.zeros(1000, dtype=int)         # 桩断面形状(0-圆形,1-方形)
        self.ksu = np.zeros(1000, dtype=int)         # 桩底约束条件
        self.agl = np.zeros((1000, 3), dtype=float)  # 桩的倾斜方向余弦
        self.nfr = np.zeros(1000, dtype=int)         # 桩地上段段数
        self.hfr = np.zeros((1000, 15), dtype=float) # 桩地上段每段高度
        self.dof = np.zeros((1000, 15), dtype=float) # 桩地上段每段直径
        self.nsf = np.zeros((1000, 15), dtype=int)   # 桩地上段计算分段数
        self.nbl = np.zeros(1000, dtype=int)         # 桩地下段段数
        self.hbl = np.zeros((1000, 15), dtype=float) # 桩地下段每段高度
        self.dob = np.zeros((1000, 15), dtype=float) # 桩地下段每段直径
        self.pmt = np.zeros((1000, 15), dtype=float) # 桩地下段每段地基反力系数
        self.pfi = np.zeros((1000, 15), dtype=float) # 桩地下段每段摩擦角
        self.nsg = np.zeros((1000, 15), dtype=int)   # 桩地下段计算分段数
        self.pmb = np.zeros(1000, dtype=float)       # 桩端土抗力系数
        self.peh = np.zeros(1000, dtype=float)       # 桩材弹性模量
        self.pke = np.zeros(1000, dtype=float)       # 桩材剪切模量与弹性模量比

        # 模拟桩信息
        self.sxy = np.zeros((20, 2), dtype=float)    # 模拟桩坐标
        self.ksctr = np.zeros(20, dtype=int)         # 模拟桩控制信息

        # 桩单元刚度
        self.esp = np.zeros((1000000, 6), dtype=float)
        
        # 输入输出文件
        self.input_file = None
        self.output_file = None
        self.pos_file = None

    def head1(self):
        """显示程序头信息"""
        print("\n\n\n\n\n")
        print("\n     Welcome to use the BCAD_PILE program !!\n")
        print("     This program is aimed to execute spatial statical analysis of pile")
        print("foundations of bridge substructures. If you have any questions about")
        print("this program, please do not hesitate to write to :\n")
        print("                                                  CAD Reseach Group")
        print("                                                  Dept.of Bridge Engr.")
        print("                                                  Tongji University")
        print("                                                  1239 Sipin Road ")
        print("                                                  Shanghai 200092")
        print("                                                  P.R.of China\n")

    def head2(self):
        """输出到文件的程序头信息"""
        self.output_file.write("\n\n\n\n\n\n")
        self.output_file.write("       ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        self.output_file.write("       +                                                                                          +\n")
        self.output_file.write("       +    BBBBBB       CCCC        A       DDDDD         PPPPPP     III     L         EEEEEEE    +\n")
        self.output_file.write("       +    B     B     C    C      A A      D    D        P     P     I      L         E          +\n")
        self.output_file.write("       +    B     B    C           A   A     D     D       P     P     I      L         E          +\n")
        self.output_file.write("       +    BBBBBB     C          A     A    D     D       PPPPPP      I      L         EEEEEEE    +\n")
        self.output_file.write("       +    B     B    C          AAAAAAA    D     D       P           I      L         E          +\n")
        self.output_file.write("       +    B     B     C    C    A     A    D    D        P           I      L     L   E          +\n")
        self.output_file.write("       +    BBBBBB       CCCC     A     A    DDDDD   ===== P          III      LLLLL    EEEEEEE    +\n")
        self.output_file.write("       +                                                                                          +\n")
        self.output_file.write("       +                        Copyright 1990, Version 1.10  modfied by Zhiq Wang                  +\n")
        self.output_file.write("       ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        self.output_file.write("\n               Welcome to use the BCAD_PILE program !!\n")
        self.output_file.write("               This program is aimed to execute spatial statical analysis of pile")
        self.output_file.write("foundations of bridge substructures. If you have any questions about")
        self.output_file.write("this program, please do not hesitate to write to :\n")
        self.output_file.write("                                                                    CAD Reseach Group\n")
        self.output_file.write("                                                                    Dept.of Bridge Engr.\n")
        self.output_file.write("                                                                    Tongji University\n")
        self.output_file.write("                                                                    1239 Sipin Road \n")
        self.output_file.write("                                                                    Shanghai 200092\n")
        self.output_file.write("                                                                    P.R.of China\n")

    def f_name(self, filename, ext):
        """组合文件名和扩展名"""
        base_name = filename.strip()
        return base_name + ext

    def r_data(self, jctr, ino, force):
        """读取初始结构数据"""
        # 读取[CONTROL]块
        title = self.input_file.readline().strip()
        jctr = int(self.input_file.readline().strip())
        
        if jctr == 1:
            nact = int(self.input_file.readline().strip())
            axy = np.zeros((10, 2), dtype=float)
            act = np.zeros((10, 6), dtype=float)
            
            for i in range(nact):
                line = self.input_file.readline().strip().split()
                axy[i, 0] = float(line[0])
                axy[i, 1] = float(line[1])
                
                line = self.input_file.readline().strip().split()
                for j in range(6):
                    act[i, j] = float(line[j])
            
            # 调用函数合并外部荷载
            force = self.init6(nact, axy, act)
        
        if jctr == 2:
            pass  # 只计算整个桩基础的刚度
        
        if jctr == 3:
            ino = int(self.input_file.readline().strip())
        
        tag = self.input_file.readline().strip()
        
        # 读取[ARRANGE]块
        title = self.input_file.readline().strip()
        line = self.input_file.readline().strip().split()
        pnum = int(line[0])
        snum = int(line[1])
        
        # 读取非模拟桩的位置坐标
        coords = []
        for k in range(pnum):
            line = self.input_file.readline().strip().split()
            if len(line) < 2:
                # 如果这行不足两个数，可能是坐标分布在多行
                while len(line) < 2:
                    more_data = self.input_file.readline().strip().split()
                    line.extend(more_data)
            
            self.pxy[k, 0] = float(line[0])
            self.pxy[k, 1] = float(line[1])
        
        # 读取模拟桩的位置坐标(如果有)
        if snum > 0:
            for k in range(snum):
                line = self.input_file.readline().strip().split()
                if len(line) < 2:
                    # 同样处理可能的多行数据
                    while len(line) < 2:
                        more_data = self.input_file.readline().strip().split()
                        line.extend(more_data)
                
                self.sxy[k, 0] = float(line[0])
                self.sxy[k, 1] = float(line[1])
        
        tag = self.input_file.readline().strip()
        
        # 读取[NO_SIMU]块
        title = self.input_file.readline().strip()
        
        # 可能的多行数据处理
        kctr_data = []
        while len(kctr_data) < pnum:
            line = self.input_file.readline().strip().split()
            kctr_data.extend(line)
        
        for k in range(pnum):
            self.kctr[k] = int(kctr_data[k])
        
        idf = self.init1(pnum, self.kctr)
        
        # 读取<0>段信息
        stag = self.input_file.readline().strip()
        if stag != "<0>":
            print(f"Error: Expected <0>, got {stag}")
            exit(1)
        
        # 初始化<0>段信息
        self.init2(0, pnum)
        
        # 读取控制信息不同的桩信息
        for ik in range(1, idf):
            line = self.input_file.readline().strip()
            # 提取<>中的数字
            if '<' in line and '>' in line:
                im = int(line.split('<')[1].split('>')[0])
            else:
                print(f"Error: Expected <number>, got {line}")
                exit(1)
            
            if im > 0:
                # 读取指定控制信息的桩
                self.init2(im, pnum)
            elif im < 0:
                # 读取需要修改的参数
                jj = int(self.input_file.readline().strip())
                sig = []
                jnew = []
                vnew = []
                
                for ia in range(jj):
                    line = self.input_file.readline().strip().split()
                    sig.append(line[0])
                    jnew.append(int(line[1]))
                    vnew.append(float(line[2]))
                
                # 修改指定控制信息的桩的参数
                self.init4(im, jj, pnum, sig, jnew, vnew)
        
        tag = self.input_file.readline().strip()
        
        # 读取[SIMUPILE]块
        title = self.input_file.readline().strip()
        if snum > 0:
            # 可能的多行数据处理
            ksctr_data = []
            while len(ksctr_data) < snum:
                line = self.input_file.readline().strip().split()
                ksctr_data.extend(line)
            
            for ks in range(snum):
                self.ksctr[ks] = int(ksctr_data[ks])
            
            idf = self.init1(snum, self.ksctr)
            is_val = pnum * 6
            
            for ik in range(idf):
                line = self.input_file.readline().strip()
                # 提取<>中的数字
                if '<' in line and '>' in line:
                    im = int(line.split('<')[1].split('>')[0])
                else:
                    print(f"Error: Expected <number>, got {line}")
                    exit(1)
                
                self.init5(im, is_val, snum)
        
        tag = self.input_file.readline().strip()
        
        # 计算桩地上和地下段长度
        zfr = np.zeros(pnum, dtype=float)
        zbl = np.zeros(pnum, dtype=float)
        
        for k in range(pnum):
            zfr[k] = 0.0
            zbl[k] = 0.0
            
            for ia in range(int(self.nfr[k])):
                zfr[k] += self.hfr[k, ia]
            
            for ia in range(int(self.nbl[k])):
                zbl[k] += self.hbl[k, ia]
        
        return jctr, ino, pnum, snum, force, zfr, zbl

    def init1(self, pnum, kctr):
        """计算控制信息不同的桩数"""
        idf = 1
        for k in range(1, pnum):
            for ki in range(k):
                if kctr[k] == kctr[ki]:
                    break
            else:
                idf += 1
        return idf

    def init2(self, im, pnum):
        """读取非模拟桩的<0>段信息"""
        line = self.input_file.readline().strip().split()
        ksh1 = int(line[0])
        ksu1 = int(line[1])
        agl1 = np.array([float(line[2]), float(line[3]), float(line[4])])
        
        # 读取地上段信息
        line = self.input_file.readline().strip().split()
        nfr1 = int(line[0])
        hfr1 = np.zeros(15, dtype=float)
        dof1 = np.zeros(15, dtype=float)
        nsf1 = np.zeros(15, dtype=int)
        
        idx = 1
        for i in range(nfr1):
            # 检查是否有足够的元素
            if idx + 2 >= len(line):
                # 读取下一行获取更多数据
                more_data = self.input_file.readline().strip().split()
                line.extend(more_data)
            
            hfr1[i] = float(line[idx])
            dof1[i] = float(line[idx+1])
            nsf1[i] = int(line[idx+2])
            idx += 3
        
        # 读取地下段信息
        line = self.input_file.readline().strip().split()
        nbl1 = int(line[0])
        hbl1 = np.zeros(15, dtype=float)
        dob1 = np.zeros(15, dtype=float)
        pmt1 = np.zeros(15, dtype=float)
        pfi1 = np.zeros(15, dtype=float)
        nsg1 = np.zeros(15, dtype=int)
        
        idx = 1
        for i in range(nbl1):
            # 检查是否有足够的元素
            if idx + 4 >= len(line):
                # 读取下一行获取更多数据
                more_data = self.input_file.readline().strip().split()
                line.extend(more_data)
            
            hbl1[i] = float(line[idx])
            dob1[i] = float(line[idx+1])
            pmt1[i] = float(line[idx+2])
            pfi1[i] = float(line[idx+3])
            nsg1[i] = int(line[idx+4])
            idx += 5
        
        # 读取桩底参数
        line = self.input_file.readline().strip().split()
        if len(line) < 3:
            # 如果这行数据不足，则读取下一行
            more_data = self.input_file.readline().strip().split()
            line.extend(more_data)
        
        pmb1 = float(line[0])
        peh1 = float(line[1])
        pke1 = float(line[2])
        
        # 将<0>段信息赋给对应控制信息的桩
        for k in range(pnum):
            ktest = self.init3(im, self.kctr[k])
            if ktest == 0:
                continue
            
            self.ksh[k] = ksh1
            self.ksu[k] = ksu1
            for ia in range(3):
                self.agl[k, ia] = agl1[ia]
            
            self.nfr[k] = nfr1
            for ii in range(nfr1):
                self.hfr[k, ii] = hfr1[ii]
                self.dof[k, ii] = dof1[ii]
                self.nsf[k, ii] = nsf1[ii]
            
            self.nbl[k] = nbl1
            for ii in range(nbl1):
                self.hbl[k, ii] = hbl1[ii]
                self.dob[k, ii] = dob1[ii]
                self.pmt[k, ii] = pmt1[ii]
                self.pfi[k, ii] = pfi1[ii]
                self.nsg[k, ii] = nsg1[ii]
            
            self.pmb[k] = pmb1
            self.peh[k] = peh1
            self.pke[k] = pke1

    def init3(self, im, k):
        """测试IM值"""
        ktest = 0
        if im == 0 and k <= 0:
            ktest = 1
        if im >= 0 and k == im:
            ktest = 1
        return ktest

    def init4(self, im, jj, pnum, sig, jnew, vnew):
        """读取<-I>段信息并修改初始信息"""
        # 找出所有控制信息为im的桩
        nim = []
        for k in range(pnum):
            if self.kctr[k] == im:
                nim.append(k)
        
        if not nim:
            print(f"Error: <{im}>")
            exit(1)
        
        # 根据标识符修改对应的参数
        for ia in range(jj):
            if sig[ia] == "KSH=":
                for k in nim:
                    self.ksh[k] = int(vnew[ia])
            elif sig[ia] == "KSU=":
                for k in nim:
                    self.ksu[k] = int(vnew[ia])
            elif sig[ia] == "AGL=":
                for k in nim:
                    self.agl[k, jnew[ia]] = vnew[ia]
            elif sig[ia] == "NFR=":
                for k in nim:
                    self.nfr[k] = int(vnew[ia])
            elif sig[ia] == "HFR=":
                for k in nim:
                    self.hfr[k, jnew[ia]] = vnew[ia]
            elif sig[ia] == "DOF=":
                for k in nim:
                    self.dof[k, jnew[ia]] = vnew[ia]
            elif sig[ia] == "NSF=":
                for k in nim:
                    self.nsf[k, jnew[ia]] = int(vnew[ia])
            elif sig[ia] == "NBL=":
                for k in nim:
                    self.nbl[k] = int(vnew[ia])
            elif sig[ia] == "HBL=":
                for k in nim:
                    self.hbl[k, jnew[ia]] = vnew[ia]
            elif sig[ia] == "DOB=":
                for k in nim:
                    self.dob[k, jnew[ia]] = vnew[ia]
            elif sig[ia] == "PMT=":
                for k in nim:
                    self.pmt[k, jnew[ia]] = vnew[ia]
            elif sig[ia] == "PFI=":
                for k in nim:
                    self.pfi[k, jnew[ia]] = vnew[ia]
            elif sig[ia] == "NSG=":
                for k in nim:
                    self.nsg[k, jnew[ia]] = int(vnew[ia])
            elif sig[ia] == "PMB=":
                for k in nim:
                    self.pmb[k] = vnew[ia]
            elif sig[ia] == "PEH=":
                for k in nim:
                    self.peh[k] = vnew[ia]
            elif sig[ia] == "PKE=":
                for k in nim:
                    self.pke[k] = vnew[ia]
            else:
                print(f"Error: <{im}>")
                exit(1)

    def init5(self, im, is_val, snum):
        """读取模拟桩信息"""
        if im < 0:
            # 读取对角元素
            line = self.input_file.readline().strip().split()
            # 处理可能的多行数据
            while len(line) < 6:
                more_data = self.input_file.readline().strip().split()
                line.extend(more_data)
            
            a = np.array([float(line[i]) for i in range(6)])
            
            for k in range(snum):
                if self.ksctr[k] == im:
                    for ia in range(6):
                        is_val += 1
                        for ib in range(6):
                            self.esp[is_val, ib] = 0.0
                        self.esp[is_val, ia] = a[ia]
        
        if im > 0:
            # 读取完整刚度矩阵
            b = np.zeros((6, 6), dtype=float)
            for ia in range(6):
                line = self.input_file.readline().strip().split()
                # 处理可能的多行数据
                while len(line) < 6:
                    more_data = self.input_file.readline().strip().split()
                    line.extend(more_data)
                
                for ib in range(6):
                    b[ia, ib] = float(line[ib])
            
            for k in range(snum):
                if self.ksctr[k] == im:
                    for ia in range(6):
                        is_val += 1
                        for ib in range(6):
                            self.esp[is_val, ib] = b[ia, ib]

    def init6(self, nact, axy, act):
        """组合外部荷载"""
        force = np.zeros(6, dtype=float)
        
        for i in range(nact):
            a = act[i, :]
            tu = self.tmatx(axy[i, 0], axy[i, 1])
            tn = tu.T
            b = np.dot(tn, a)
            force += b
        
        return force

    def btxy(self, pnum, zfr, zbl, btx, bty):
        """计算桩的变形系数"""
        # 计算桩在地面处的坐标
        gxy = np.zeros((pnum, 2), dtype=float)
        for k in range(pnum):
            gxy[k, 0] = self.pxy[k, 0] + zfr[k] * self.agl[k, 0]
            gxy[k, 1] = self.pxy[k, 1] + zfr[k] * self.agl[k, 1]
        
        # 计算桩间距
        for k in range(pnum):
            for k1 in range(k+1, pnum):
                s = np.sqrt((gxy[k, 0] - gxy[k1, 0])**2 + (gxy[k, 1] - gxy[k1, 1])**2) - (self.dob[k, 0] + self.dob[k1, 0]) / 2.0
                if s < 1.0:
                    # 桩间距小于1m，调用kinf1函数
                    kinf = np.zeros(2, dtype=float)
                    self.kinf1(0, pnum, self.dob, zbl, gxy, kinf, 0)
                    self.kinf1(1, pnum, self.dob, zbl, gxy, kinf, 1)
                    break
        else:
            # 桩间距大于1m，调用kinf2函数
            kinf = np.zeros(2, dtype=float)
            self.kinf2(0, pnum, self.dob, zbl, gxy, kinf, 0)
            self.kinf2(1, pnum, self.dob, zbl, gxy, kinf, 1)
        
        # 计算每个桩的变形系数
        for k in range(pnum):
            if k > 0:
                # 检查是否有相同控制信息的桩，如果有则复制变形系数
                for k1 in range(k):
                    if self.kctr[k] == self.kctr[k1]:
                        for ia in range(int(self.nbl[k1])):
                            btx[k, ia] = btx[k1, ia]
                            bty[k, ia] = bty[k1, ia]
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
                        btx[k, ia] = (self.pmt[k, ia] * bx1 / (self.peh[k] * b))**0.2
                        bty[k, ia] = (self.pmt[k, ia] * by1 / (self.peh[k] * b))**0.2
            else:
                # 计算第一个桩的变形系数
                ka = 1.0
                if self.ksh[k] == 1:
                    ka = 0.9
                
                for ia in range(int(self.nbl[k])):
                    bx1 = ka * kinf[0] * (self.dob[k, ia] + 1.0)
                    by1 = ka * kinf[1] * (self.dob[k, ia] + 1.0)
                    a, b = self.eaj(self.ksh[k], self.pke[k], self.dob[k, ia])
                    btx[k, ia] = (self.pmt[k, ia] * bx1 / (self.peh[k] * b))**0.2
                    bty[k, ia] = (self.pmt[k, ia] * by1 / (self.peh[k] * b))**0.2

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

    def area(self, pnum, zfr, zbl, ao):
        """计算桩底面积"""
        # 计算桩底坐标
        bxy = np.zeros((pnum, 2), dtype=float)
        w = np.zeros(pnum, dtype=float)
        smin = np.ones(pnum, dtype=float) * 100.0
        
        for k in range(pnum):
            bxy[k, 0] = self.pxy[k, 0] + (zfr[k] + zbl[k]) * self.agl[k, 0]
            bxy[k, 1] = self.pxy[k, 1] + (zfr[k] + zbl[k]) * self.agl[k, 1]
            
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
        for k in range(pnum):
            for ia in range(k+1, pnum):
                s = math.sqrt((bxy[k, 0] - bxy[ia, 0])**2 + (bxy[k, 1] - bxy[ia, 1])**2)
                if s < smin[k]:
                    smin[k] = s
                if s < smin[ia]:
                    smin[ia] = s
        
        # 确定使用最小宽度并计算桩底面积
        for k in range(pnum):
            if smin[k] < w[k]:
                w[k] = smin[k]
            
            if self.ksh[k] == 0:  # 圆形
                ao[k] = math.pi * w[k]**2 / 4.0
            else:  # 方形
                ao[k] = w[k]**2

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

    def stiff_n(self, pnum, zfr, zbl, ao, rzz):
        """计算每个桩的轴向刚度"""
        # 计算第一个桩的轴向刚度
        self.stn(0, zbl[0], ao[0], rzz[0:1])
        
        # 计算其他桩的轴向刚度
        for k in range(1, pnum):
            # 检查是否有相同控制信息和底面积的桩
            for ia in range(k):
                if self.kctr[k] == self.kctr[ia] and ao[k] == ao[ia]:
                    rzz[k] = rzz[ia]
                    break
            else:
                # 计算新桩的轴向刚度
                self.stn(k, zbl[k], ao[k], rzz[k:k+1])

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

    def pstiff(self, pnum, rzz, btx, bty):
        """计算桩单元刚度"""
        for k in range(pnum):
            # 如果桩无地下段，使用单位矩阵
            if self.nbl[k] == 0:
                kbx = np.eye(4)
                kby = np.eye(4)
            else:
                # 收集桩段信息
                h = np.zeros(100, dtype=float)
                bt1 = np.zeros(15, dtype=float)
                bt2 = np.zeros(15, dtype=float)
                ej = np.zeros(15, dtype=float)
                
                h[0] = 0.0
                for ia in range(int(self.nbl[k])):
                    bt1[ia] = btx[k, ia]
                    bt2[ia] = bty[k, ia]
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
                h = np.zeros(100, dtype=float)
                ej = np.zeros(15, dtype=float)
                
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
            self.cndtn(self.ksu[k], kx, ky, rzz[k], ke)
            
            # 保存桩的单元刚度
            for i in range(6):
                for j in range(6):
                    self.esp[(k-1)*6+i, j] = ke[i, j]

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

    def disp(self, jctr, ino, pnum, snum, force, duk, so):
        """计算桩基础帽的位移"""
        # 计算整个桩基础的刚度
        so.fill(0.0)
        
        for k in range(pnum + snum):
            # 获取桩的单元刚度
            a = np.zeros((6, 6), dtype=float)
            for i in range(6):
                for j in range(6):
                    a[i, j] = self.esp[(k-1)*6+i, j]
            
            # 应用转换矩阵
            if k < pnum:
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
                x = self.sxy[k-pnum, 0]
                y = self.sxy[k-pnum, 1]
            
            # 应用位置转换矩阵
            tu = np.zeros((6, 6), dtype=float)
            self.tmatx(x, y, tu)
            tn = tu.T
            b = np.dot(a, tu)
            a = np.dot(tn, b)
            
            # 累加到整体刚度矩阵
            so += a
        
        # 只计算指定桩的刚度
        if jctr == 3:
            self.output_file.write(f"\n\n       *** Stiffness of the No.{ino} pile ***\n\n")
            for i in range(6):
                line = "       " + " ".join([f"{self.esp[(ino-1)*6+i, j]:12.4e}" for j in range(6)])
                self.output_file.write(line + "\n")
            return so
        
        # 只计算整个桩基础的刚度
        if jctr == 2:
            self.output_file.write("\n\n       *** Stiffness of the entire pile foundation ***\n\n")
            for i in range(6):
                line = "       " + " ".join([f"{so[i, j]:12.4e}" for j in range(6)])
                self.output_file.write(line + "\n")
            return so
        
        # 求解位移
        force = np.linalg.solve(so, force)
        
        # 输出位移结果
        self.output_file.write("\n       *****************************************************************************************************\n")
        self.output_file.write("               DISPLACEMENTS AT THE CAP CENTER OF PILE FOUNDATION\n")
        self.output_file.write("       *****************************************************************************************************\n")
        self.output_file.write(f"\n                Movement in the direction of X axis : UX= {force[0]:12.4e} (m)\n")
        self.output_file.write(f"                Movement in the direction of Y axis : UY= {force[1]:12.4e} (m)\n")
        self.output_file.write(f"                Movement in the direction of Z axis : UZ= {force[2]:12.4e} (m)\n")
        self.output_file.write(f"                Rotational angle  around X axis :     SX= {force[3]:12.4e} (rad)\n")
        self.output_file.write(f"                Rotational angle around Y axis :      SY= {force[4]:12.4e} (rad)\n")
        self.output_file.write(f"                Rotational angle around Z axis :      SZ= {force[5]:12.4e} (rad)\n\n")
        
        # 计算每个桩的局部位移
        for k in range(pnum):
            # 应用位置转换矩阵
            tu = np.zeros((6, 6), dtype=float)
            self.tmatx(self.pxy[k, 0], self.pxy[k, 1], tu)
            c1 = np.dot(tu, force)
            
            # 应用方向转换矩阵
            tk = np.zeros((6, 6), dtype=float)
            self.trnsfr(self.agl[k, 0], self.agl[k, 1], self.agl[k, 2], tk)
            tk1 = tk.T
            c = np.dot(tk1, c1)
            
            # 保存桩的局部位移
            for i in range(6):
                duk[k, i] = c[i]
        
        return so

    def eforce(self, pnum, btx, bty, zbl, duk):
        """计算桩体的位移和内力"""
        # 输出桩的平面位置到pos文件
        self.pos_file.write(f"{pnum}\n")
        for i in range(pnum):
            line = " ".join([f"{self.pxy[i, j]:14.4e}" for j in range(2)])
            self.pos_file.write(line + "\n")
        
        # 计算每个桩的位移和内力
        for k in range(pnum):
            # 获取桩的单元刚度和位移
            ce = duk[k, :]
            se = np.zeros((6, 6), dtype=float)
            for i in range(6):
                for j in range(6):
                    se[i, j] = self.esp[(k-1)*6+i, j]
            
            # 计算桩顶部的内力
            pe = np.dot(se, ce)
            
            # 初始化数组
            max_nodes = 100
            zh = np.zeros(max_nodes, dtype=float)
            fx = np.zeros((max_nodes, 4), dtype=float)
            fy = np.zeros((max_nodes, 4), dtype=float)
            fz = np.zeros(max_nodes, dtype=float)
            psx = np.zeros(max_nodes, dtype=float)
            psy = np.zeros(max_nodes, dtype=float)
            
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
                    self.saa(btx[k, ia], ej, h1, h2, r)
                    xb = np.dot(r, xa)
                    
                    if abs(btx[k, ia] - bty[k, ia]) > 1.0e-3:
                        r = np.zeros((4, 4), dtype=float)
                        self.saa(bty[k, ia], ej, h1, h2, r)
                    
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
                        fz[nsum] = fz[ig] * (1.0 - h2**2 / zbl[k]**2)
            
            # 输出桩体位移和内力到输出文件
            self.output_file.write("       ****************************************************************************************\n")
            self.output_file.write(f"                                  NO. {k} # PILE\n")
            self.output_file.write("       ****************************************************************************************\n")
            self.output_file.write(f"\n            Coordinator of the pile: (x,y) = ({self.pxy[k, 0]:12.4e} ,{self.pxy[k, 1]:12.4e} )\n\n")
            self.output_file.write("            Displacements and internal forces at the top of pile:\n")
            self.output_file.write(f"\n               UX= {ce[0]:12.4e} (m)         NX= {pe[0]:12.4e} (t)\n")
            self.output_file.write(f"               UY= {ce[1]:12.4e} (m)         NY= {pe[1]:12.4e} (t)\n")
            self.output_file.write(f"               UZ= {ce[2]:12.4e} (m)         NZ= {pe[2]:12.4e} (t)\n")
            self.output_file.write(f"               SX= {ce[3]:12.4e} (rad)       MX= {pe[3]:12.4e} (t*m)\n")
            self.output_file.write(f"               SY= {ce[4]:12.4e} (rad)       MY= {pe[4]:12.4e} (t*m)\n")
            self.output_file.write(f"               SZ= {ce[5]:12.4e} (rad)       MZ= {pe[5]:12.4e} (t*m)\n\n")
            
            self.output_file.write("       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            self.output_file.write("                                Displacements of the pile body and\n")
            self.output_file.write("                             Compression stresses of soil (PSX,PSY)\n")
            self.output_file.write("       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            self.output_file.write("\n               Z            UX            UY            SX            SY            PSX            PSY\n")
            self.output_file.write("              (m)           (m)           (m)          (rad)         (rad)         (t/m2)         (t/m2)\n\n")
            
            # 输出地上段数据
            for i in range(ig):
                line = f"       {zh[i]:14.4e}{fx[i, 0]:14.4e}{fy[i, 0]:14.4e}{fy[i, 1]:14.4e}{fx[i, 1]:14.4e}"
                self.output_file.write(line + "\n")
            
            # 输出地下段数据
            for i in range(ig, nsum+1):
                line = f"       {zh[i]:14.4e}{fx[i, 0]:14.4e}{fy[i, 0]:14.4e}{fy[i, 1]:14.4e}{fx[i, 1]:14.4e}{psx[i]:14.4e}{psy[i]:14.4e}"
                self.output_file.write(line + "\n")
            
            self.output_file.write("\n\n")
            self.output_file.write("       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            self.output_file.write("                                Internal forces of the pile body\n")
            self.output_file.write("       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            self.output_file.write("\n                  Z              NX              NY              NZ              MX              MY\n")
            self.output_file.write("                 (m)             (t)             (t)             (t)            (t*m)           (t*m)\n\n")
            
            # 输出内力数据
            for i in range(nsum+1):
                line = f"       {zh[i]:16.4e}{fx[i, 2]:16.4e}{fy[i, 2]:16.4e}{fz[i]:16.4e}{fy[i, 3]:16.4e}{fx[i, 3]:16.4e}"
                self.output_file.write(line + "\n")
            
            # 输出位移和内力数据到pos文件
            # 为地上段设置零土压力
            for i in range(ig):
                psx[i] = 0.0
                psy[i] = 0.0
            
            # 写入桩号和节点数
            self.pos_file.write(f"{k} {nsum+1}\n")
            # 写入桩顶坐标
            self.pos_file.write(f"{self.pxy[k, 0]:14.4e} {self.pxy[k, 1]:14.4e}\n")
            # 写入各节点Z坐标
            line = " ".join([f"{zh[i]:14.4e}" for i in range(nsum+1)])
            self.pos_file.write(line + "\n")
            # 写入X方向变形和内力
            for i in range(nsum+1):
                line = " ".join([f"{fx[i, j]:14.4e}" for j in range(4)])
                self.pos_file.write(line + "\n")
            # 写入Y方向变形和内力
            for i in range(nsum+1):
                line = " ".join([f"{fy[i, j]:14.4e}" for j in range(4)])
                self.pos_file.write(line + "\n")
            # 写入Z方向内力
            line = " ".join([f"{fz[i]:14.4e}" for i in range(nsum+1)])
            self.pos_file.write(line + "\n")
            # 写入X方向土压力
            line = " ".join([f"{psx[i]:14.4e}" for i in range(nsum+1)])
            self.pos_file.write(line + "\n")
            # 写入Y方向土压力
            line = " ".join([f"{psy[i]:14.4e}" for i in range(nsum+1)])
            self.pos_file.write(line + "\n")

    def run(self):
        """运行程序的主函数"""
        # 显示程序头
        self.head1()
        
        # 读取数据文件名
        print("\n       Please enter data filename:")
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
        jctr = 0
        ino = 0
        force = np.zeros(6, dtype=float)
        jctr, ino, pnum, snum, force, zfr, zbl = self.r_data(jctr, ino, force)
        
        # 计算桩的变形因子
        print("\n\n       *** To calculate deformation factors of piles ***")
        btx = np.zeros((pnum, 15), dtype=float)
        bty = np.zeros((pnum, 15), dtype=float)
        self.btxy(pnum, zfr, zbl, btx, bty)
        
        # 计算桩底面积和轴向刚度
        print("\n\n       *** To calculate axis stiffness of piles ***")
        ao = np.zeros(pnum, dtype=float)
        self.area(pnum, zfr, zbl, ao)
        rzz = np.zeros(pnum, dtype=float)
        self.stiff_n(pnum, zfr, zbl, ao, rzz)
        
        # 计算桩的侧向刚度
        print("\n\n       *** To calculate lateral stiffness of piles ***")
        self.pstiff(pnum, rzz, btx, bty)
        
        # 计算桩基础帽的位移
        print("\n\n       *** To execute entire pile foundation analysis ***\n\n")
        duk = np.zeros((pnum, 6), dtype=float)
        so = np.zeros((6, 6), dtype=float)
        self.disp(jctr, ino, pnum, snum, force, duk, so)
        
        # 计算桩体的位移和内力
        self.eforce(pnum, btx, bty, zbl, duk)
        
        # 关闭文件
        self.input_file.close()
        self.output_file.close()
        self.pos_file.close()
        
        print("\n程序运行完成，结果已保存到 %s 和 %s 文件中。" % (output_filename, pos_filename))

if __name__ == "__main__":
    bcad = BCADPile()
    bcad.run()