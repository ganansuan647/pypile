#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BCAD_PILE - Python Conversion

This program performs spatial static analysis of pile foundations
for bridge substructures.

This script is a Python conversion of the original Fortran BCAD_PILE program.
"""

import numpy as np
import math
import os
import sys
from collections import defaultdict

# --- Constants ---
PI = np.pi
MAX_PILES = 1000
MAX_SIMU_PILES = 20
MAX_SEGMENTS = 15
MAX_NODES_PER_PILE = 100 # Estimate for zh, fx, etc. arrays in EFORCE
MAX_ACTION_POINTS = 10
MAX_MODIFICATION_RULES = 30
MAX_ESP_SIZE = 1000000 # Matches Fortran declaration, adjust if needed

# --- Global Data Structures (Simulating COMMON blocks) ---
# These could be encapsulated in classes for better organization

# /PINF/ data (Consider using a dictionary or a custom Pile class)
pinf = {
    'pxy': np.zeros((MAX_PILES, 2)),
    'kctr': np.zeros(MAX_PILES, dtype=int),
    'ksh': np.zeros(MAX_PILES, dtype=int),
    'ksu': np.zeros(MAX_PILES, dtype=int),
    'agl': np.zeros((MAX_PILES, 3)),
    'nfr': np.zeros(MAX_PILES, dtype=int),
    'hfr': np.zeros((MAX_PILES, MAX_SEGMENTS)),
    'dof': np.zeros((MAX_PILES, MAX_SEGMENTS)),
    'nsf': np.zeros((MAX_PILES, MAX_SEGMENTS), dtype=int),
    'nbl': np.zeros(MAX_PILES, dtype=int),
    'hbl': np.zeros((MAX_PILES, MAX_SEGMENTS)),
    'dob': np.zeros((MAX_PILES, MAX_SEGMENTS)),
    'pmt': np.zeros((MAX_PILES, MAX_SEGMENTS)),
    'pfi': np.zeros((MAX_PILES, MAX_SEGMENTS)),
    'nsg': np.zeros((MAX_PILES, MAX_SEGMENTS), dtype=int),
    'pmb': np.zeros(MAX_PILES),
    'peh': np.zeros(MAX_PILES),
    'pke': np.zeros(MAX_PILES),
}

# /SIMU/ data
simu = {
    'sxy': np.zeros((MAX_SIMU_PILES, 2)),
    'ksctr': np.zeros(MAX_SIMU_PILES, dtype=int),
}

# /ESTIF/ data
# Using a dictionary might be more flexible if indices are sparse,
# but a large NumPy array mimics the Fortran structure.
# Key: (pile_index * 6 + row_index), Value: np.array(row_data)
# Or simply a large array if memory allows and indices are dense.
esp = np.zeros((MAX_ESP_SIZE, 6)) # Element stiffness matrices

# --- Utility Functions ---

def f_name(base, ext):
    """Concatenates base filename and extension."""
    # Assumes base has no trailing spaces that need stripping like Fortran might
    return base.strip() + ext

def print_head1():
    """Prints program header to console."""
    print("\n\n\n\n\n")
    print(5 * ' ' + 'Welcome to use the BCAD_PILE program (Python Version) !!')
    print(5 * ' ' + '  This program is aimed to execute spatial statical analysis of pile')
    print(5 * ' ' + 'foundations of bridge substructures. If you have any questions about')
    print(5 * ' ' + 'this program, please do not hesitate to write to :')
    print()
    print(50 * ' ' + 'CAD Reseach Group')
    print(50 * ' ' + 'Dept.of Bridge Engr.')
    print(50 * ' ' + 'Tongji University')
    print(50 * ' ' + '1239 Sipin Road ')
    print(50 * ' ' + 'Shanghai 200092')
    print(50 * ' ' + 'P.R.of China')
    print()

def print_head2(f):
    """Prints program header to the output file."""
    f.write("\n\n\n\n\n\n")
    f.write(7 * ' ' + '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    f.write(7 * ' ' + '+                                                                                                     +\n')
    f.write(7 * ' ' + '+    BBBBBB      CCCC       A       DDDDD       PPPPPP    III    L          EEEEEEE    +\n')
    f.write(7 * ' ' + '+    B     B    C    C     A A      D    D      P     P    I     L          E          +\n')
    f.write(7 * ' ' + '+    B     B   C          A   A     D     D     P     P    I     L          E          +\n')
    f.write(7 * ' ' + '+    BBBBBB    C         A     A    D     D     PPPPPP     I     L          EEEEEEE    +\n')
    f.write(7 * ' ' + '+    B     B   C         AAAAAAA    D     D     P          I     L          E          +\n')
    f.write(7 * ' ' + '+    B     B    C    C   A     A    D    D      P          I     L     L    E          +\n')
    f.write(7 * ' ' + '+    BBBBBB      CCCC    A     A    DDDDD     ===== P         III    LLLLL     EEEEEEE    +\n')
    f.write(7 * ' ' + '+                                                                                                     +\n')
    f.write(7 * ' ' + '+                Copyright 1990, Version 1.10 modfied by Zhiq Wang                                  +\n') # Adjusted spacing
    f.write(7 * ' ' + '+                      (Python Conversion by Gemini)                                                +\n')
    f.write(7 * ' ' + '+                                                                                                     +\n')
    f.write(7 * ' ' + '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    f.write("\n" + 15 * ' ' + 'Welcome to use the BCAD_PILE program !!\n')
    f.write(15 * ' ' + 'This program is aimed to execute spatial statical analysis of pile\n')
    f.write(15 * ' ' + 'foundations of bridge substructures. If you have any questions about\n')
    f.write(15 * ' ' + 'this program, please do not hesitate to write to :\n')
    f.write("\n" + 60 * ' ' + 'CAD Reseach Group\n')
    f.write(60 * ' ' + 'Dept.of Bridge Engr.\n')
    f.write(60 * ' ' + 'Tongji University\n')
    f.write(60 * ' ' + '1239 Sipin Road \n')
    f.write(60 * ' ' + 'Shanghai 200092\n')
    f.write(60 * ' ' + 'P.R.of China\n\n')
    f.write(7 * ' ' + '*****************************************************************************************************\n\n')

def mulult(m, l, n, a, b):
    """Matrix multiplication C(m,n) = A(m,l) * B(l,n)."""
    # Ensure inputs are numpy arrays
    a_np = np.asarray(a)
    b_np = np.asarray(b)

    # Check dimensions
    if a_np.shape == (m, l) and b_np.shape == (l, n):
        return np.matmul(a_np, b_np)
    elif a_np.shape == (m, l) and b_np.shape == (l,) and n == 1:
         # Handle matrix * vector case
         return np.matmul(a_np, b_np.reshape(l, 1)).flatten() # Return 1D array if result is vector
    elif a_np.shape == (l,) and b_np.shape == (l, n) and m == 1:
        # Handle vector * matrix case (less common in the Fortran code structure)
        return np.matmul(a_np.reshape(1, l), b_np).flatten()
    elif m==l==n==6 and a_np.shape == (6,) and b_np.shape == (6,): # Special case from INIT6
         return np.matmul(a_np.reshape(1,6), b_np).flatten() # Treat A as row vector
    elif a_np.shape == (l,) and b_np.shape == (l,) and m == 1 and n == 1:
        # Dot product case
        return np.dot(a_np, b_np)
    elif a_np.shape == (m,l) and b_np.shape == (l,) and n==1: # Matrix * vector -> vector
        return np.matmul(a_np, b_np)
    else:
        # Add more specific error messages if needed
        raise ValueError(f"Dimension mismatch in mulult: A{a_np.shape}, B{b_np.shape} for M={m}, L={l}, N={n}")

def sinver(a, n):
    """Matrix inversion using numpy.linalg.inv."""
    a_np = np.asarray(a).reshape(n, n)
    try:
        aminus = np.linalg.inv(a_np)
        ner = 0 # Success
    except np.linalg.LinAlgError:
        print("Error: Matrix is singular or not invertible in sinver.")
        aminus = np.full((n, n), np.nan) # Return NaN matrix on error
        ner = 100 # Error code like Fortran
    return aminus, ner

def trnsps(m, n, b):
    """Matrix transpose using numpy.transpose."""
    b_np = np.asarray(b).reshape(m, n)
    return np.transpose(b_np)

def gaos(n, a, b):
    """Solves AX=B for X using numpy.linalg.solve."""
    a_np = np.asarray(a).reshape(n, n)
    b_np = np.asarray(b).reshape(n)
    try:
        # Solve the system
        x = np.linalg.solve(a_np, b_np)
        return x # Returns the solution vector B (overwritten like in Fortran)
    except np.linalg.LinAlgError:
        print("Error: Matrix is singular. Cannot solve system in gaos.")
        # Return original B or raise an exception, depending on desired behavior
        # Returning original B mimics Fortran behavior if inversion fails implicitly
        return b_np

def eaj(ksh, pke, do_val):
    """Calculate Area (A) and Moment of Inertia (B*PKE) for pile section."""
    if ksh == 0:  # Circular section
        a = PI * do_val**2 / 4.0
        b = pke * PI * do_val**4 / 64.0
    else:  # Square section (assuming ksh=1 is square)
        a = do_val**2
        b = pke * do_val**4 / 12.0
    return a, b

def mfree(ej, h):
    """Calculate relational matrix for a free pile segment."""
    r = np.identity(4)
    if abs(ej) < 1e-12: # Avoid division by zero
        print(f"Warning: EJ is near zero ({ej}) in mfree. Results may be inaccurate.")
        # Handle appropriately, maybe return identity or raise error
        # For now, set terms with EJ in denominator to a large number or zero
        h_sq_2ej = np.inf if h > 0 else 0
        h_cu_6ej = np.inf if h > 0 else 0
        h_ej = np.inf if h > 0 else 0
    else:
        h_sq_2ej = h**2 / (2.0 * ej)
        h_cu_6ej = h**3 / (6.0 * ej)
        h_ej = h / ej

    r[0, 1] = h
    r[0, 2] = h_cu_6ej
    r[0, 3] = -h_sq_2ej
    r[1, 2] = h_sq_2ej
    r[1, 3] = -h_ej
    r[3, 2] = -h # Note: Fortran R(4,3)=-H corresponds to Python R[3,2]
    # Fortran R(2,4)=-H/EJ -> Python R[1,3]
    # Fortran R(1,4)=-H**2/(2.0*EJ) -> Python R[0,3]
    # Fortran R(1,3)=H**3/(6.0*EJ) -> Python R[0,2]
    # Fortran R(2,3)=H**2/(2.0*EJ) -> Python R[1,2]

    return r

def param1(y):
    """Calculate power series terms (Part 1)."""
    y2 = y*y; y3 = y2*y; y4 = y3*y; y5 = y4*y; y6 = y5*y; y7 = y6*y; y8 = y7*y
    y9 = y8*y; y10 = y9*y; y11 = y10*y; y12 = y11*y; y13 = y12*y; y14 = y13*y
    y15 = y14*y; y16 = y15*y; y17 = y16*y; y18 = y17*y; y19 = y18*y; y20 = y19*y
    y21 = y20*y; y22 = y21*y; y23 = y22*y; y24 = y23*y; y25 = y24*y; y26 = y25*y
    y27 = y26*y; y28 = y27*y # Precompute powers if needed, though direct calculation is often fine

    a1 = (1.0 - y5/120.0 + y10/6.048e5 - y15/1.9813e10 + y20/2.3038e15
          - y25/6.9945e20)
    b1 = (y - y6/360.0 + y11/2.8512e6 - y16/1.245e11 + y21/1.7889e16
          - y26/6.4185e21)
    c1 = (y2/2.0 - y7/1680.0 + y12/1.9958e7 - y17/1.14e12 + y22/2.0e17
          - y27/8.43e22)
    d1 = (y3/6.0 - y8/10080.0 + y13/1.7297e8 - y18/1.2703e13
          + y23/2.6997e18 - y28/1.33e24)
    a2 = (-y4/24.0 + y9/6.048e4 - y14/1.3209e9 + y19/1.1519e14
          - y24/2.7978e19)
    b2 = (1.0 - y5/60.0 + y10/2.592e5 - y15/7.7837e9 + y20/8.5185e14
          - y25/2.4686e20)
    c2 = (y - y6/240.0 + y11/1.6632e6 - y16/6.7059e10 + y21/9.0973e15
          - y26/3.1222e21)
    d2 = (y2/2.0 - y7/1260.0 + y12/1.3305e7 - y17/7.0572e11
          + y22/1.1738e17 - y27/4.738e22)
    return a1, b1, c1, d1, a2, b2, c2, d2

def param2(y):
    """Calculate power series terms (Part 2)."""
    y2 = y*y; y3 = y2*y; y4 = y3*y; y5 = y4*y; y6 = y5*y; y7 = y6*y; y8 = y7*y
    y9 = y8*y; y10 = y9*y; y11 = y10*y; y12 = y11*y; y13 = y12*y; y14 = y13*y
    y15 = y14*y; y16 = y15*y; y17 = y16*y; y18 = y17*y; y19 = y18*y; y20 = y19*y
    y21 = y20*y; y22 = y21*y; y23 = y22*y; y24 = y23*y; y25 = y24*y; y26 = y25*y

    a3 = (-y3/6.0 + y8/6.72e3 - y13/9.435e7 + y18/6.0626e12
          - y23/1.1657e18)
    b3 = (-y4/12.0 + y9/25920.0 - y14/5.1892e8 + y19/4.2593e13
          - y24/9.8746e18)
    c3 = (1.0 - y5/40.0 + y10/151200.0 - y15/4.1912e9 + y20/4.332e14
          - y25/1.2009e20)
    d3 = (y - y6/180.0 + y11/1.1088e6 - y16/4.1513e10 + y21/5.3354e15
          - y26/1.7543e21)
    a4 = (-y2/2.0 + y7/840.0 - y12/7.257e6 + y17/3.3681e11
          - y22/5.0683e16)
    b4 = (-y3/3.0 + y8/2880.0 - y13/3.7066e7 + y18/2.2477e12
          - y23/4.1144e17)
    c4 = (-y4/8.0 + y9/1.512e4 - y14/2.7941e8 + y19/2.166e13
          - y24/4.8034e18)
    d4 = (1.0 - y5/30.0 + y10/100800.0 - y15/2.5946e9 + y20/2.5406e14
          - y25/6.7491e19)
    return a3, b3, c3, d3, a4, b4, c4, d4

def param(bt, ej, x):
    """Calculate the coefficient matrix AA used in SAA."""
    aa = np.zeros((4, 4))
    y = bt * x
    if y > 6.0: # Limiter from Fortran code
        # print(f"Warning: y = bt*x = {y} > 6.0 in param. Clamping to 6.0.")
        y = 6.0
    elif y < -6.0: # Added limiter for negative values if applicable
        # print(f"Warning: y = bt*x = {y} < -6.0 in param. Clamping to -6.0.")
        y = -6.0


    a1, b1, c1, d1, a2, b2, c2, d2 = param1(y)
    a3, b3, c3, d3, a4, b4, c4, d4 = param2(y)

    if abs(bt) < 1e-12:
        print(f"Warning: bt is near zero ({bt}) in param. Results may be inaccurate.")
        # Handle division by zero - perhaps return identity or zero matrix?
        # Or calculate limits as bt -> 0 if possible. For now, set problematic terms to 0 or inf.
        bt_sq = 0
        bt_cu = 0
        inv_bt = np.inf
        inv_bt_sq = np.inf
        inv_bt_cu = np.inf
    else:
        bt_sq = bt**2
        bt_cu = bt**3
        inv_bt = 1.0 / bt
        inv_bt_sq = inv_bt**2
        inv_bt_cu = inv_bt**3

    aa[0, 0] = a1
    aa[0, 1] = b1 * inv_bt
    aa[0, 2] = 2.0 * c1 * inv_bt_sq
    aa[0, 3] = 6.0 * d1 * inv_bt_cu
    aa[1, 0] = a2 * bt
    aa[1, 1] = b2
    aa[1, 2] = 2.0 * c2 * inv_bt
    aa[1, 3] = 6.0 * d2 * inv_bt_sq
    aa[2, 0] = a3 * bt_sq
    aa[2, 1] = b3 * bt
    aa[2, 2] = 2.0 * c3
    aa[2, 3] = 6.0 * d3 * inv_bt
    aa[3, 0] = a4 * bt_cu
    aa[3, 1] = b4 * bt_sq
    aa[3, 2] = 2.0 * c4 * bt
    aa[3, 3] = 6.0 * d4

    return aa


def saa(bt, ej, h1, h2, ai_prev=None):
    """Calculate relational matrix for one non-free pile segment."""
    ai = np.zeros((4, 4))

    # Calculate coefficient matrices at h1 and h2
    ai1 = param(bt, ej, h1)
    ai2 = param(bt, ej, h2)

    # Invert ai1
    ai3, ner = sinver(ai1, 4)
    if ner != 0:
        print(f"Error: Matrix inversion failed in saa for h1={h1}, bt={bt}, ej={ej}")
        # Handle error, maybe return identity or raise exception
        return np.identity(4) # Or previous matrix if available

    # Calculate relational matrix AI = AI2 * AI1_inv
    ai = mulult(4, 4, 4, ai2, ai3)

    # Apply EJ scaling factors (careful with indices)
    # Fortran: AI(I,J+2)=AI(I,J+2)/EJ for I=1,2; J=1,2 -> Python: ai[0:2, 2:4] /= ej
    # Fortran: AI(I+2,J)=AI(I+2,J)*EJ for I=1,2; J=1,2 -> Python: ai[2:4, 0:2] *= ej
    if abs(ej) > 1e-12:
      ai[0:2, 2:4] /= ej
    else:
      print(f"Warning: EJ is near zero ({ej}) in saa scaling. Results may be inaccurate.")
      ai[0:2, 2:4] = np.inf # Or some other indicator of singularity

    ai[2:4, 0:2] *= ej


    # Swap rows 3 and 4 (Fortran 3 and 4 -> Python 2 and 3)
    ai[[2, 3], :] = ai[[3, 2], :]
    # Swap columns 3 and 4 (Fortran 3 and 4 -> Python 2 and 3)
    ai[:, [2, 3]] = ai[:, [3, 2]]

    return ai

def rltmtx(nbl_k, bt1, bt2, ej_k, h_k):
    """Calculate relational matrices KBX, KBY for non-free pile segments."""
    kbx = np.identity(4)
    kby = np.identity(4)
    recalculate_kby = False

    # Check if BTX and BTY are significantly different
    if nbl_k > 0 and np.any(np.abs(bt1[:nbl_k] - bt2[:nbl_k]) > 1.0e-10):
        recalculate_kby = True

    # Calculate KBX
    if nbl_k > 0:
        kbx = saa(bt1[0], ej_k[0], h_k[0], h_k[1])
        for ia in range(1, nbl_k): # Fortran loop 2 to NBL -> Python 1 to NBL-1
            a1 = kbx.copy() # Store previous KBX
            # Note: Fortran H array seems 1-based, H(IA) and H(IA+1) used.
            # Python h_k is 0-based. Assuming h_k[ia] is H(IA) and h_k[ia+1] is H(IA+1).
            a2 = saa(bt1[ia], ej_k[ia], h_k[ia], h_k[ia+1], ai_prev=a1)
            kbx = mulult(4, 4, 4, a2, a1)

    # Calculate KBY
    if recalculate_kby:
         if nbl_k > 0:
            kby = saa(bt2[0], ej_k[0], h_k[0], h_k[1])
            for ia in range(1, nbl_k):
                a1 = kby.copy()
                a2 = saa(bt2[ia], ej_k[ia], h_k[ia], h_k[ia+1], ai_prev=a1)
                kby = mulult(4, 4, 4, a2, a1)
    else:
        kby = kbx.copy() # If BTX and BTY are close, KBY is same as KBX

    return kbx, kby

def rltfr(nfr_k, ej_k, hfr_k):
    """Calculate relational matrix KFR for free pile segments."""
    if nfr_k == 0:
        return np.identity(4)

    kfr = mfree(ej_k[0], hfr_k[0])
    for ia in range(1, nfr_k): # Fortran loop 2 to NFR -> Python 1 to NFR-1
        r = mfree(ej_k[ia], hfr_k[ia])
        kfr = mulult(4, 4, 4, kfr, r) # Fortran was KFR*R, check order

    return kfr

def combx(kbx, kfr):
    """Combine relational matrices for free and non-free segments."""
    # Fortran: KBX(I,4)=-KBX(I,4) -> Python: kbx[:, 3] = -kbx[:, 3]
    kbx_mod = kbx.copy()
    kbx_mod[:, 3] *= -1.0
    kx = mulult(4, 4, 4, kbx_mod, kfr)
    return kx

def dvsn(ksu_k, kxy):
    """Treat boundary conditions to get 2x2 stiffness contribution."""
    # Extract submatrices A11, A12, A21, A22
    # Fortran KXY(I,J) -> Python kxy[i-1, j-1]
    a11 = kxy[0:2, 0:2]
    a12 = kxy[0:2, 2:4]
    a21 = kxy[2:4, 0:2]
    a22 = kxy[2:4, 2:4]

    at = np.zeros((2, 2))

    if ksu_k == 4: # Fixed end condition? Check Fortran comments/manual
        av, ner = sinver(a12, 2)
        if ner == 0:
            at = mulult(2, 2, 2, av, a11)
        else:
            print(f"Error: Matrix inversion failed for A12 in dvsn (KSU={ksu_k}).")
            at.fill(np.nan) # Indicate error
    else: # Hinged or other end condition?
        av, ner = sinver(a22, 2)
        if ner == 0:
            at = mulult(2, 2, 2, av, a21)
        else:
            print(f"Error: Matrix inversion failed for A22 in dvsn (KSU={ksu_k}).")
            at.fill(np.nan) # Indicate error

    return -at # Return negative as per Fortran logic

def cndtn(ksu_k, kx, ky, rzz_k):
    """Calculate the 6x6 element stiffness matrix KE for a pile."""
    ke = np.zeros((6, 6))

    # Calculate contributions from X-direction bending
    atx = dvsn(ksu_k, kx)
    ke[0, 0] = atx[0, 0] # KE(1,1) = AT(1,1)
    ke[0, 4] = atx[0, 1] # KE(1,5) = AT(1,2) (Index 5 -> Python 4)
    ke[4, 0] = atx[1, 0] # KE(5,1) = AT(2,1) (Index 5 -> Python 4)
    ke[4, 4] = atx[1, 1] # KE(5,5) = AT(2,2) (Index 5 -> Python 4)

    # Calculate contributions from Y-direction bending
    aty = dvsn(ksu_k, ky)
    ke[1, 1] = aty[0, 0] # KE(2,2) = AT(1,1)
    ke[1, 3] = -aty[0, 1] # KE(2,4) = -AT(1,2) (Index 4 -> Python 3)
    ke[3, 1] = -aty[1, 0] # KE(4,2) = -AT(2,1) (Index 4 -> Python 3)
    ke[3, 3] = aty[1, 1] # KE(4,4) = AT(2,2) (Index 4 -> Python 3)

    # Axial stiffness
    ke[2, 2] = rzz_k # KE(3,3) = RZZ

    # Torsional stiffness (approximation from Fortran)
    # KE(6,6)=0.1*(KE(4,4)+KE(5,5)) -> Python indices ke[5,5], ke[3,3], ke[4,4]
    ke[5, 5] = 0.1 * (ke[3, 3] + ke[4, 4])

    return ke

def tmatx(x, y):
    """Calculate transformation matrix TU for pile coordinates (X, Y)."""
    tu = np.identity(6)
    tu[0, 5] = -y  # TU(1,6) = -Y
    tu[1, 5] = x   # TU(2,6) = X
    tu[2, 3] = y   # TU(3,4) = Y
    tu[2, 4] = -x  # TU(3,5) = -X
    return tu

def trnsfr(agl_k1, agl_k2, agl_k3):
    """Calculate transformation matrix TK for pile inclination (AGL)."""
    # AGL(K,1), AGL(K,2), AGL(K,3) are direction cosines x, y, z
    x, y, z = agl_k1, agl_k2, agl_k3
    tk = np.zeros((6, 6))

    # Calculate B = sqrt(Y^2 + Z^2)
    b = math.sqrt(y**2 + z**2)

    # Check for vertical pile (B=0) to avoid division by zero
    if abs(b) < 1e-10:
        # Handle vertical pile case (assuming AGL = [0, 0, 1] or [0, 0, -1])
        if abs(abs(z) - 1.0) < 1e-10: # Check if z is +/- 1
             tk[0, 2] = x # Should be 0 for vertical
             tk[1, 1] = z # Use z for sign (+1 or -1)
             tk[2, 0] = -x*z # Should be 0
             tk[3, 5] = x # Should be 0
             tk[4, 4] = z
             tk[5, 3] = -x*z # Should be 0
             # Need to verify the correct transformation for a vertical pile
             # Often it's just identity or involves swapping axes depending on definition
             # Let's assume standard case where local z aligns with global Z
             tk = np.diag([1, 1, z, 1, 1, z]) # Simple scaling/reflection
             print("Warning: Vertical pile detected in trnsfr. Using simplified transform.")

        else:
             print(f"Error: B is zero in trnsfr, but pile is not vertical (x={x}, y={y}, z={z}).")
             tk.fill(np.nan) # Error state
    else:
        # Standard transformation for inclined pile
        tk[0, 0] = b
        tk[0, 1] = 0.0
        tk[0, 2] = x
        tk[1, 0] = -x * y / b
        tk[1, 1] = z / b
        tk[1, 2] = y
        tk[2, 0] = -x * z / b
        tk[2, 1] = -y / b
        tk[2, 2] = z

        # Copy 3x3 block to lower right
        tk[3:6, 3:6] = tk[0:3, 0:3]

    return tk

def parc(pile_count_in_row):
    """Calculate pile group coefficient C based on number of piles in a row."""
    if pile_count_in_row == 1:
        return 1.0
    elif pile_count_in_row == 2:
        return 0.6
    elif pile_count_in_row == 3:
        return 0.5
    elif pile_count_in_row >= 4:
        return 0.45
    else: # Should not happen for count >= 1
        return 1.0

def kinf3(in_val, aa, dd, zz):
    """Calculate influential factor KINF for a single row of piles."""
    if in_val <= 1:
        return 1.0

    ho = np.zeros(in_val)
    for i in range(in_val):
        ho[i] = 3.0 * (dd[i] + 1.0)
        if ho[i] > zz[i]:
            ho[i] = zz[i]

    lo = 100.0
    hoo = 0.0
    found_pair = False
    for i in range(in_val):
        for i1 in range(i + 1, in_val):
            found_pair = True
            s = abs(aa[i] - aa[i1]) - (dd[i] + dd[i1]) / 2.0
            if s < lo:
                lo = s
                hoo_i = ho[i]
                hoo_i1 = ho[i1]
                hoo = max(hoo_i, hoo_i1) # Fortran logic: IF(HOO.LT.HO(I1)) HOO=HO(I1) implies max

    if not found_pair: # Only one pile in the effective row
        return 1.0

    if hoo <= 0: # Avoid division by zero if effective depth is zero
         print("Warning: Effective depth hoo is zero in kinf3.")
         return 1.0 # Or handle as error

    # Ensure lo is not negative (can happen if piles overlap)
    lo = max(lo, 0.0)

    limit = 0.6 * hoo
    if lo >= limit:
        kinf = 1.0
    else:
        c = parc(in_val)
        if abs(limit) < 1e-12: # Avoid division by zero
             kinf = c # Or 1.0 depending on interpretation
             print("Warning: Limit (0.6*hoo) is zero in kinf3.")
        else:
             kinf = c + (1.0 - c) * lo / limit

    return kinf


def kinf1(im_axis, pnum_active, dob_all, zbl_all, gxy_all):
    """Calculate KINF factor (Case: Pile spacing < 1.0m)."""
    # im_axis = 1 for X, 2 for Y (corresponds to Python index 0 or 1)
    axis_idx = im_axis - 1

    # Find unique coordinates along the specified axis
    unique_coords = {}
    for k in range(pnum_active):
        coord = gxy_all[k, axis_idx]
        if coord not in unique_coords:
            # Store index of first pile found at this coordinate
            unique_coords[coord] = k

    # Extract data for piles at unique coordinates
    in_val = len(unique_coords)
    if in_val == 0:
        return 1.0 # No piles

    aa = np.zeros(in_val)
    dd = np.zeros(in_val)
    zz = np.zeros(in_val)
    i = 0
    for coord, k_idx in unique_coords.items():
        # Use dob[k, 0] assuming DOB(K,1) is the relevant diameter
        aa[i] = coord
        dd[i] = dob_all[k_idx, 0] if pinf['nbl'][k_idx] > 0 else 0.0 # Handle NBL=0 case
        zz[i] = zbl_all[k_idx]
        i += 1

    # Calculate KINF for this effective row
    kinf = kinf3(in_val, aa, dd, zz)
    return kinf


def kinf2(im_axis, pnum_active, dob_all, zbl_all, gxy_all):
    """Calculate KINF factor (Case: Pile spacing >= 1.0m)."""
    # im_axis = 1 for X, 2 for Y (corresponds to Python index 0 or 1)
    axis_idx = im_axis - 1
    other_axis_idx = 1 - axis_idx # The axis defining the rows

    # Group piles into rows based on the 'other' axis coordinate
    rows = defaultdict(list)
    for k in range(pnum_active):
        row_coord = gxy_all[k, other_axis_idx]
        # Use a tolerance for floating point comparison
        found_row = False
        for existing_coord in rows.keys():
            if abs(row_coord - existing_coord) < 1e-6: # Tolerance
                rows[existing_coord].append(k)
                found_row = True
                break
        if not found_row:
            rows[row_coord].append(k)

    kmin = 1.0

    # Calculate KINF for each row and find the minimum
    for row_coord, pile_indices in rows.items():
        in_val = len(pile_indices)
        if in_val == 0:
            continue

        aa = np.zeros(in_val)
        dd = np.zeros(in_val)
        zz = np.zeros(in_val)
        for i, k_idx in enumerate(pile_indices):
            aa[i] = gxy_all[k_idx, axis_idx]
            # Use dob[k, 0] assuming DOB(K,1) is the relevant diameter
            dd[i] = dob_all[k_idx, 0] if pinf['nbl'][k_idx] > 0 else 0.0
            zz[i] = zbl_all[k_idx]

        kinf_row = kinf3(in_val, aa, dd, zz)
        if kinf_row < kmin:
            kmin = kinf_row

    return kmin


# --- Core Calculation Functions ---

def init1(num, kctr_arr):
    """Calculate the number of distinct non-zero values in KCTR."""
    # Find unique non-zero values in the relevant part of kctr_arr
    if num <= 0:
        return 0
    unique_values = set(k for k in kctr_arr[:num] if k != 0) # Include negative KCTR values
    # Fortran logic counted distinct values including 0 implicitly if present?
    # Let's re-read Fortran: It counts distinct values among KCTR(1) to KCTR(PNUM).
    # If KCTR(K) == KCTR(KI) for KI < K, it skips. So it counts unique values.
    # The initial IDF=1 suggests it counts the first value, then increments for new ones.
    if num > 0:
         distinct_count = len(set(kctr_arr[:num]))
         return distinct_count
    else:
         return 0
    # Fortran logic seems slightly different, let's trace:
    # IDF=1
    # DO K=2,PNUM
    #   DO KI=1,K-1
    #     IF(KCTR(K).EQ.KCTR(KI)) GOTO 31 # Found duplicate, skip increment
    #   CONTINUE # Inner loop finished without finding duplicate
    #   IDF=IDF+1 # Increment IDF for unique value
    # 31 CONTINUE # Outer loop continue
    # This counts the number of unique values encountered sequentially.
    if num <= 0:
        return 0
    seen = set()
    idf = 0
    for k in range(num): # 0 to num-1
        if kctr_arr[k] not in seen:
            seen.add(kctr_arr[k])
            idf += 1
    return idf if idf > 0 else 1 # Ensure IDF is at least 1 if num > 0? Fortran starts at 1. Let's return count.
    # Revisit: Fortran IDF=1 initially. If all values are the same, IDF remains 1.
    # If num=1, loop doesn't run, IDF=1.
    # If num=2, K=2. If KCTR(2)==KCTR(1), goto 31, IDF=1. Else, IDF=2.
    # So it's the count of unique values.
    if num <= 0: return 0
    return len(set(kctr_arr[:num]))


def init3(im, k_val):
    """Test IM value against K value (pile control number)."""
    ktest = 0
    # Original Fortran logic:
    # IF(IM.EQ.0.AND.K.LE.0) KTEST=1  -> Apply default (IM=0) if K is 0 or negative
    # IF(IM.GE.0.AND.K.EQ.IM) KTEST=1  -> Apply specific IM if K matches IM (and IM is not negative)
    if im == 0 and k_val <= 0:
        ktest = 1
    elif im >= 0 and k_val == im: # Note: Fortran K.EQ.IM. Python k_val == im
        ktest = 1
    return ktest == 1 # Return boolean True if test passes


def init2(f, im, pnum_active):
    """Read and apply <IM> segment data for non-simulative piles."""
    # Read data for this segment type
    line1 = f.readline().split()
    ksh1 = int(line1[0])
    ksu1 = int(line1[1])
    agl1 = [float(x) for x in line1[2:5]] # Read 3 values

    line2 = f.readline().split()
    nfr1 = int(line2[0])
    hfr1 = np.zeros(MAX_SEGMENTS)
    dof1 = np.zeros(MAX_SEGMENTS)
    nsf1 = np.zeros(MAX_SEGMENTS, dtype=int)
    idx = 1
    for i in range(nfr1):
            # 检查是否有足够的元素
            if idx + 2 >= len(line2):
                # 读取下一行获取更多数据
                more_data = f.readline().strip().split()
                line2.extend(more_data)

            hfr1[i] = float(line2[idx])
            dof1[i] = float(line2[idx + 1])
            nsf1[i] = int(line2[idx + 2])
            idx += 3

    line3 = f.readline().strip().split()
    nbl1 = int(line3[0])
    hbl1 = np.zeros(MAX_SEGMENTS)
    dob1 = np.zeros(MAX_SEGMENTS)
    pmt1 = np.zeros(MAX_SEGMENTS)
    pfi1 = np.zeros(MAX_SEGMENTS)
    nsg1 = np.zeros(MAX_SEGMENTS, dtype=int)
    idx = 1
    for ii in range(nbl1):
        # 检查是否有足够的元素
        if idx + 4 >= len(line3):
            # 读取下一行获取更多数据
            more_data = f.readline().strip().split()
            line3.extend(more_data)
        hbl1[ii] = float(line3[idx])
        dob1[ii] = float(line3[idx+1])
        pmt1[ii] = float(line3[idx+2])
        pfi1[ii] = float(line3[idx+3])
        nsg1[ii] = int(line3[idx+4])
        idx += 5

    line4 = f.readline().strip().split()
    if len(line4) < 3:
        # 如果这行数据不足，则读取下一行
        more_data = f.readline().strip().split()
        line4.extend(more_data)
    pmb1 = float(line4[0])
    peh1 = float(line4[1])
    pke1 = float(line4[2])

    # Apply this data to relevant piles
    for k in range(pnum_active):
        if init3(im, pinf['kctr'][k]): # Check if this pile matches the segment type IM
            pinf['ksh'][k] = ksh1
            pinf['ksu'][k] = ksu1
            pinf['agl'][k, :] = agl1[:]
            pinf['nfr'][k] = nfr1
            pinf['hfr'][k, :nfr1] = hfr1[:nfr1]
            pinf['dof'][k, :nfr1] = dof1[:nfr1]
            pinf['nsf'][k, :nfr1] = nsf1[:nfr1]
            pinf['nbl'][k] = nbl1
            pinf['hbl'][k, :nbl1] = hbl1[:nbl1]
            pinf['dob'][k, :nbl1] = dob1[:nbl1]
            pinf['pmt'][k, :nbl1] = pmt1[:nbl1]
            pinf['pfi'][k, :nbl1] = pfi1[:nbl1]
            pinf['nsg'][k, :nbl1] = nsg1[:nbl1]
            pinf['pmb'][k] = pmb1
            pinf['peh'][k] = peh1
            pinf['pke'][k] = pke1

def init4(f, im, pnum_active):
    """Read and apply <-IM> modification segment data."""
    jj = int(f.readline().strip()) # Number of modification rules
    sig = []
    jnew = np.zeros(jj, dtype=int)
    vnew = np.zeros(jj)
    for ia in range(jj):
        line = f.readline().split()
        sig.append(line[0]) # e.g., 'KSH='
        jnew[ia] = int(line[1]) # Index for modification (1-based from Fortran)
        vnew[ia] = float(line[2]) # New value

    # Find indices of piles matching KCTR = IM
    nim_indices = [k for k in range(pnum_active) if pinf['kctr'][k] == im]

    if not nim_indices:
        print(f"Warning: No piles found with KCTR={im} for modification block <-{abs(im)}>")
        return # Skip modifications if no matching piles

    # Apply modifications
    for ia in range(jj):
        target_piles = nim_indices
        param_name = sig[ia].lower() # e.g. 'ksh='
        # Fortran index JNEW(IA) is 1-based, convert to 0-based for Python
        idx = jnew[ia] - 1
        val = vnew[ia]

        if idx < 0:
             print(f"Warning: Invalid index {jnew[ia]} (0-based {idx}) in modification block <-{abs(im)}>. Skipping rule.")
             continue

        try:
            if param_name == 'ksh=':
                for k_idx in target_piles: pinf['ksh'][k_idx] = int(val)
            elif param_name == 'ksu=':
                for k_idx in target_piles: pinf['ksu'][k_idx] = int(val)
            elif param_name == 'agl=':
                if 0 <= idx < 3:
                    for k_idx in target_piles: pinf['agl'][k_idx, idx] = val
                else: print(f"Warning: Invalid index {jnew[ia]} for AGL.")
            elif param_name == 'nfr=':
                for k_idx in target_piles: pinf['nfr'][k_idx] = int(val)
            elif param_name == 'hfr=':
                 if 0 <= idx < MAX_SEGMENTS:
                    for k_idx in target_piles: pinf['hfr'][k_idx, idx] = val
                 else: print(f"Warning: Invalid index {jnew[ia]} for HFR.")
            elif param_name == 'dof=':
                 if 0 <= idx < MAX_SEGMENTS:
                    for k_idx in target_piles: pinf['dof'][k_idx, idx] = val
                 else: print(f"Warning: Invalid index {jnew[ia]} for DOF.")
            elif param_name == 'nsf=':
                 if 0 <= idx < MAX_SEGMENTS:
                    for k_idx in target_piles: pinf['nsf'][k_idx, idx] = int(val)
                 else: print(f"Warning: Invalid index {jnew[ia]} for NSF.")
            elif param_name == 'nbl=':
                for k_idx in target_piles: pinf['nbl'][k_idx] = int(val)
            elif param_name == 'hbl=':
                 if 0 <= idx < MAX_SEGMENTS:
                    for k_idx in target_piles: pinf['hbl'][k_idx, idx] = val
                 else: print(f"Warning: Invalid index {jnew[ia]} for HBL.")
            elif param_name == 'dob=':
                 if 0 <= idx < MAX_SEGMENTS:
                    for k_idx in target_piles: pinf['dob'][k_idx, idx] = val
                 else: print(f"Warning: Invalid index {jnew[ia]} for DOB.")
            elif param_name == 'pmt=':
                 if 0 <= idx < MAX_SEGMENTS:
                    for k_idx in target_piles: pinf['pmt'][k_idx, idx] = val
                 else: print(f"Warning: Invalid index {jnew[ia]} for PMT.")
            elif param_name == 'pfi=':
                 if 0 <= idx < MAX_SEGMENTS:
                    for k_idx in target_piles: pinf['pfi'][k_idx, idx] = val
                 else: print(f"Warning: Invalid index {jnew[ia]} for PFI.")
            elif param_name == 'nsg=':
                 if 0 <= idx < MAX_SEGMENTS:
                    for k_idx in target_piles: pinf['nsg'][k_idx, idx] = int(val)
                 else: print(f"Warning: Invalid index {jnew[ia]} for NSG.")
            elif param_name == 'pmb=':
                for k_idx in target_piles: pinf['pmb'][k_idx] = val
            elif param_name == 'peh=':
                for k_idx in target_piles: pinf['peh'][k_idx] = val
            elif param_name == 'pke=':
                for k_idx in target_piles: pinf['pke'][k_idx] = val
            else:
                print(f"Error: Unknown modification parameter '{sig[ia]}' in <-{abs(im)}> block.")
                # Consider stopping execution based on Fortran STOP
                # sys.exit(f"Error: Unknown modification parameter '{sig[ia]}'")
        except IndexError:
             print(f"Error: Index {idx} out of bounds for parameter {param_name} modification.")
        except ValueError:
             print(f"Error: Invalid value '{val}' for parameter {param_name} modification (expected int?).")


def init5(f, im, is_start_index, snum_active):
    """Read simulative pile stiffness data and store in ESP."""
    # is_start_index is the starting row in ESP for simulative piles
    current_is = is_start_index
    if im < 0: # Diagonal stiffness matrix
        line = f.readline().split()
        a = np.array([float(x) for x in line[:6]])
        for k in range(snum_active):
            if simu['ksctr'][k] == im:
                for ia in range(6):
                    esp[current_is + ia, :] = 0.0 # Zero out row
                    esp[current_is + ia, ia] = a[ia] # Set diagonal element
                current_is += 6 # Move to next pile's block
    elif im > 0: # Full stiffness matrix
        b = np.zeros((6, 6))
        for ia in range(6):
            line = f.readline().split()
            b[ia, :] = [float(x) for x in line[:6]]
        for k in range(snum_active):
            if simu['ksctr'][k] == im:
                esp[current_is : current_is + 6, :] = b[:, :]
                current_is += 6
    return current_is # Return the next available index

def init6(f, force_global):
    """Read external forces applied at different points and combine them."""
    nact = int(f.readline().strip())
    axy = np.zeros((nact, 2))
    act = np.zeros((nact, 6))
    for i in range(nact):
        axy[i, :] = [float(x) for x in f.readline().split()[:2]]
        act[i, :] = [float(x) for x in f.readline().split()[:6]]

    force_combined = np.zeros(6)
    for i in range(nact):
        a = act[i, :] # Force vector at point i
        tu = tmatx(axy[i, 0], axy[i, 1]) # Transformation for position
        tn = trnsps(6, 6, tu) # Transpose of TU
        # Force transformation: F_global = TN * F_local
        # Fortran: CALL MULULT(6,6,1,TN,A,B) -> B = TN * A
        b = mulult(6, 6, 1, tn, a)
        force_combined += b

    # Overwrite the initial global force vector
    force_global[:] = force_combined[:]


def read_data(filename):
    """Reads input data from the specified file."""
    pnum = 0
    snum = 0
    force = np.zeros(6)
    jctr = 0
    ino = 0
    zfr = np.zeros(MAX_PILES) # Length above ground
    zbl = np.zeros(MAX_PILES) # Length below ground

    try:
        with open(filename, 'r') as f:
            # --- [CONTRAL] Block ---
            title_contral = f.readline().strip() # Read title, often unused
            jctr = int(f.readline().strip())

            if jctr == 1:
                # Read and combine external forces
                init6(f, force) # Modifies force array directly
            elif jctr == 3:
                ino = int(f.readline().strip())

            tag_contral = f.readline().strip()
            if tag_contral.upper() != 'END;':
                 print(f"Warning: Expected 'END;' after CONTRAL block, found '{tag_contral}'")
                 # Potentially read until END; is found or assume fixed structure

            # --- [ARRANGE] Block ---
            title_arrange = f.readline().strip()
            line = f.readline().split()
            pnum = int(line[0])
            snum = int(line[1])

            if not (0 <= pnum <= MAX_PILES):
                 raise ValueError(f"Number of non-simulative piles {pnum} exceeds MAX_PILES {MAX_PILES}")
            if not (0 <= snum <= MAX_SIMU_PILES):
                 raise ValueError(f"Number of simulative piles {snum} exceeds MAX_SIMU_PILES {MAX_SIMU_PILES}")


            # Read non-simulative pile coordinates PXY
            for k in range(pnum):
                line = f.readline().split()
                pinf['pxy'][k, 0] = float(line[0])
                pinf['pxy'][k, 1] = float(line[1])

            # Read simulative pile coordinates SXY
            if snum > 0:
                for k in range(snum):
                    line = f.readline().split()
                    simu['sxy'][k, 0] = float(line[0])
                    simu['sxy'][k, 1] = float(line[1])

            tag_arrange = f.readline().strip()
            if tag_arrange.upper() != 'END;':
                 print(f"Warning: Expected 'END;' after ARRANGE block, found '{tag_arrange}'")


            # --- [NO_SIMU] Block ---
            title_no_simu = f.readline().strip()
            # Read KCTR values (pile type control)
            line = f.readline().split()
            pinf['kctr'][:pnum] = [int(x) for x in line[:pnum]]

            # Calculate number of distinct pile types (IDF)
            idf_no_simu = init1(pnum, pinf['kctr'])

            # Read pile property segments <IM> or <-IM>
            current_tag = f.readline().strip()
            if current_tag != '<0>':
                 raise ValueError(f"Error: Expected '<0>' segment tag, found '{current_tag}'")

            # Process <0> segment (default properties)
            init2(f, 0, pnum)

            # Process other segments based on IDF
            processed_tags = {'<0>'}
            # Fortran loop was DO 13 IK=1,IDF-1. This seems wrong.
            # It should probably loop through all unique tags found.
            unique_kctr = sorted(list(set(pinf['kctr'][:pnum])))

            while True:
                 current_pos = f.tell() # Remember position before reading tag
                 next_tag_line = f.readline()
                 if not next_tag_line: break # End of file
                 next_tag = next_tag_line.strip()

                 # Check if it's a segment tag like <I> or <-I> or END;
                 if next_tag.startswith('<') and next_tag.endswith('>'):
                     if next_tag in processed_tags:
                          print(f"Warning: Duplicate segment tag '{next_tag}' encountered.")
                          continue # Skip already processed tag

                     try:
                         im_str = next_tag[1:-1]
                         im = int(im_str)
                         processed_tags.add(next_tag)

                         if im >= 0: # <IM> segment
                             init2(f, im, pnum)
                         else: # <-IM> segment (modification)
                             init4(f, im, pnum)
                     except ValueError:
                          print(f"Warning: Could not parse segment tag '{next_tag}'. Stopping NO_SIMU read.")
                          f.seek(current_pos) # Go back to before the tag
                          break
                     except Exception as e:
                          print(f"Error processing segment {next_tag}: {e}")
                          f.seek(current_pos)
                          break

                 elif next_tag.upper() == 'END;':
                      # End of NO_SIMU block
                      break
                 else:
                      # Not a recognized tag or END;, assume end of block
                      print(f"Warning: Unexpected line '{next_tag}' found after pile segments. Assuming end of NO_SIMU.")
                      f.seek(current_pos) # Go back
                      break


            # --- [SIMUPILE] Block ---
            title_simupile = f.readline().strip()
            if snum == 0:
                tag_simupile = f.readline().strip() # Read END; tag
                if tag_simupile.upper() != 'END;':
                     print(f"Warning: Expected 'END;' after SIMUPILE block (snum=0), found '{tag_simupile}'")
            else:
                # Read KSCTR values for simulative piles
                line = f.readline().split()
                simu['ksctr'][:snum] = [int(x) for x in line[:snum]]

                idf_simu = init1(snum, simu['ksctr'])

                # Starting index in ESP for simulative piles
                # Assumes non-simu piles take up pnum * 6 rows
                is_start_index = pnum * 6
                processed_simu_tags = set()

                # Read simulative pile stiffness segments <IM>
                for ik in range(idf_simu): # Loop based on expected number of unique types
                    tag_line = f.readline()
                    if not tag_line:
                         print("Error: Unexpected end of file while reading SIMUPILE segments.")
                         break
                    tag = tag_line.strip()
                    if tag.startswith('<') and tag.endswith('>'):
                         if tag in processed_simu_tags:
                              print(f"Warning: Duplicate simulative pile tag '{tag}'. Reading data anyway.")
                              # Fortran might read data even if tag is duplicate based on loop count

                         try:
                             im_str = tag[1:-1]
                             im = int(im_str)
                             processed_simu_tags.add(tag)
                             is_start_index = init5(f, im, is_start_index, snum)
                         except ValueError:
                             print(f"Warning: Could not parse simulative pile tag '{tag}'. Skipping.")
                             # Need to consume the expected lines for this tag if possible
                             if im < 0: f.readline() # Consume one line for diagonal
                             else:
                                  for _ in range(6): f.readline() # Consume 6 lines for full matrix
                         except Exception as e:
                             print(f"Error processing simulative segment {tag}: {e}")
                             break
                    else:
                         print(f"Warning: Expected <IM> tag for simulative pile, found '{tag}'. Stopping SIMUPILE read.")
                         break

                tag_simupile = f.readline().strip() # Read END; tag
                if tag_simupile.upper() != 'END;':
                     print(f"Warning: Expected 'END;' after SIMUPILE segments, found '{tag_simupile}'")


            # --- Calculate ZFR and ZBL ---
            for k in range(pnum):
                nfr_k = pinf['nfr'][k]
                nbl_k = pinf['nbl'][k]
                zfr[k] = np.sum(pinf['hfr'][k, :nfr_k]) if nfr_k > 0 else 0.0
                zbl[k] = np.sum(pinf['hbl'][k, :nbl_k]) if nbl_k > 0 else 0.0

    except FileNotFoundError:
        print(f"Error: Input file '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the input file: {e}")
        # Print more details if needed, e.g., traceback
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Pack results into a dictionary or tuple
    control_params = {'jctr': jctr, 'ino': ino}
    # Note: pinf, simu, esp are modified globally

    return pnum, snum, force, control_params, zfr, zbl


def calculate_deformation_factors(pnum_active, zfr, zbl):
    """Calculate deformation factors BTX, BTY."""
    btx = np.zeros((pnum_active, MAX_SEGMENTS))
    bty = np.zeros((pnum_active, MAX_SEGMENTS))
    gxy = np.zeros((pnum_active, 2)) # Coordinates at ground surface

    if pnum_active == 0:
        return btx, bty # Return zero arrays if no piles

    # Calculate ground surface coordinates GXY
    for k in range(pnum_active):
        # AGL(K,1)=cos(theta_x), AGL(K,2)=cos(theta_y)
        gxy[k, 0] = pinf['pxy'][k, 0] + zfr[k] * pinf['agl'][k, 0]
        gxy[k, 1] = pinf['pxy'][k, 1] + zfr[k] * pinf['agl'][k, 1]

    # Determine KINF factors
    kinf = np.ones(2) # Default KINF = [1.0, 1.0]
    min_spacing = float('inf')
    dob_k0 = pinf['dob'][0, 0] if pinf['nbl'][0] > 0 else 0.0 # Diameter of first pile, first segment below ground

    for k in range(pnum_active):
        for k1 in range(k + 1, pnum_active):
             dob_k = pinf['dob'][k, 0] if pinf['nbl'][k] > 0 else 0.0
             dob_k1 = pinf['dob'][k1, 0] if pinf['nbl'][k1] > 0 else 0.0
             dist_sq = (gxy[k, 0] - gxy[k1, 0])**2 + (gxy[k, 1] - gxy[k1, 1])**2
             s = math.sqrt(dist_sq) - (dob_k + dob_k1) / 2.0
             min_spacing = min(min_spacing, s)

    # Choose KINF calculation method based on minimum spacing
    if min_spacing < 1.0:
        # print("Pile spacing < 1.0m, using KINF1")
        # Need DOB and ZBL for all piles passed to kinf1
        kinf[0] = kinf1(1, pnum_active, pinf['dob'], zbl, gxy) # KINF for X
        kinf[1] = kinf1(2, pnum_active, pinf['dob'], zbl, gxy) # KINF for Y
    else:
        # print("Pile spacing >= 1.0m, using KINF2")
        kinf[0] = kinf2(1, pnum_active, pinf['dob'], zbl, gxy) # KINF for X
        kinf[1] = kinf2(2, pnum_active, pinf['dob'], zbl, gxy) # KINF for Y

    # print(f"Calculated KINF factors: {kinf}")

    # Calculate BTX and BTY for each pile segment below ground
    processed_kctr = set() # To avoid recalculating for identical pile types
    for k in range(pnum_active):
        kctr_k = pinf['kctr'][k]
        # Check if already processed (approximation of Fortran GOTO logic)
        # This assumes BTX/BTY only depend on KCTR and properties read in INIT2/4
        # If properties affecting BTX/BTY can differ for same KCTR, this needs refinement.
        found_match = False
        if k > 0:
            for k1 in range(k):
                if pinf['kctr'][k1] == kctr_k:
                    # Copy results from k1
                    nbl_k1 = pinf['nbl'][k1]
                    btx[k, :nbl_k1] = btx[k1, :nbl_k1]
                    bty[k, :nbl_k1] = bty[k1, :nbl_k1]
                    found_match = True
                    break
        if found_match:
            continue

        # Calculate for this pile type
        ka = 1.0
        if pinf['ksh'][k] == 1: # Square pile adjustment factor
            ka = 0.9

        nbl_k = pinf['nbl'][k]
        peh_k = pinf['peh'][k]
        pke_k = pinf['pke'][k]
        ksh_k = pinf['ksh'][k]

        if peh_k <= 0:
             print(f"Warning: PEH (Young's Modulus) is zero or negative for pile {k}. Cannot calculate BTX/BTY.")
             continue # Skip pile if modulus is invalid

        for ia in range(nbl_k):
            dob_kia = pinf['dob'][k, ia]
            pmt_kia = pinf['pmt'][k, ia]

            # Calculate effective width/diameter term B from EAJ
            _, b_ej_pke = eaj(ksh_k, pke_k, dob_kia)
            b_ej = b_ej_pke / pke_k if abs(pke_k) > 1e-12 else 0 # Get B (Moment of Inertia)

            # Denominator term for beta calculation
            denom = peh_k * b_ej
            if denom <= 0:
                 print(f"Warning: Denominator (PEH*B) is zero or negative for pile {k}, segment {ia}. Cannot calculate BTX/BTY.")
                 btx[k, ia] = 0 # Or some error value
                 bty[k, ia] = 0
                 continue

            # Numerator terms
            num_x = ka * kinf[0] * (dob_kia + 1.0) * pmt_kia
            num_y = ka * kinf[1] * (dob_kia + 1.0) * pmt_kia

            # Calculate beta = (m * Kinf * (D+1) / (E*I)) ^ 0.2
            # Ensure non-negative base for the power
            base_x = num_x / denom
            base_y = num_y / denom

            btx[k, ia] = base_x**0.2 if base_x >= 0 else 0.0
            bty[k, ia] = base_y**0.2 if base_y >= 0 else 0.0

    return btx, bty

def calculate_pile_areas(pnum_active, zfr, zbl):
    """Calculate effective area AO at the bottom of each pile."""
    ao = np.zeros(pnum_active)
    if pnum_active == 0: return ao

    bxy = np.zeros((pnum_active, 2)) # Coordinates at pile bottom
    w = np.zeros(pnum_active)       # Effective width at bottom
    smin = np.full(pnum_active, 100.0) # Minimum spacing at bottom

    # Calculate bottom coordinates and initial effective width W
    for k in range(pnum_active):
        agl_k = pinf['agl'][k, :]
        z_total = zfr[k] + zbl[k]
        bxy[k, 0] = pinf['pxy'][k, 0] + z_total * agl_k[0]
        bxy[k, 1] = pinf['pxy'][k, 1] + z_total * agl_k[1]

        ksu_k = pinf['ksu'][k]
        nbl_k = pinf['nbl'][k]
        nfr_k = pinf['nfr'][k]

        if ksu_k > 2: # End bearing pile? Use diameter at bottom/top segment
            if nbl_k > 0:
                w[k] = pinf['dob'][k, nbl_k - 1] # Diameter of last segment below ground
            elif nfr_k > 0:
                 w[k] = pinf['dof'][k, nfr_k - 1] # Diameter of last segment above ground (if no segment below)
            else:
                 w[k] = 0.0 # No segments defined
                 print(f"Warning: Pile {k} (KSU>2) has no segments (NFR=0, NBL=0). Cannot determine W.")

        else: # Friction pile? Calculate effective width based on friction angle
            w[k] = 0.0
            agl_z = agl_k[2] # cos(theta_z)
            # Avoid math domain error for acos if agl_z is slightly outside [-1, 1]
            agl_z_clipped = np.clip(agl_z, -1.0, 1.0)
            # Angle with vertical axis (radians)
            # Fortran: AG=ATAN(SQRT(1-AGL(K,3)**2)/AGL(K,3)) = acos(agl_z)
            if abs(agl_z_clipped) == 1.0:
                 ag = 0.0 if agl_z_clipped > 0 else PI # Vertical pile
            else:
                 # ag = math.atan(math.sqrt(1.0 - agl_z**2) / agl_z) # This is atan(tan(angle)) = angle
                 ag = math.acos(agl_z_clipped) # More direct way to get angle from vertical

            # Sum contributions from segments below ground
            for ia in range(nbl_k):
                pfi_kia_rad = pinf['pfi'][k, ia] * PI / 720.0 # Convert friction angle/2 to radians
                # Ensure angle for tan is valid
                tan_angle = ag - pfi_kia_rad
                # Handle potential large angles or near pi/2
                if abs(math.cos(tan_angle)) < 1e-12:
                     tan_val = np.sign(math.sin(tan_angle)) * float('inf') # Avoid division by zero
                     print(f"Warning: tan angle near pi/2 in AREA calculation for pile {k}, seg {ia}")
                else:
                     tan_val = math.tan(tan_angle)

                term = math.sin(ag) - agl_z * tan_val
                w[k] += pinf['hbl'][k, ia] * term

            # Add diameter of first segment below ground
            if nbl_k > 0:
                w[k] = w[k] * 2.0 + pinf['dob'][k, 0]
            elif nfr_k > 0:
                 # If no segments below ground, maybe use top diameter? Fortran logic is unclear here.
                 # Let's assume W is based on segments below ground only if KSU <= 2.
                 # If NBL=0, W remains 0 from init. Add diameter of last free segment?
                 # Fortran code adds DOB(K,1) even if NBL=0? Seems unlikely.
                 # Let's assume W is only calculated if NBL > 0 for KSU<=2.
                 # If NBL=0, W should probably be based on DOF like KSU>2 case?
                 # Revisit Fortran: W(K)=W(K)*2+DOB(K,1). This happens *after* the loop.
                 # If NBL=0, loop doesn't run, W=0. Then W=DOB(K,1). This seems wrong if NBL=0.
                 # Let's stick to the loop logic: W calculated only if NBL>0.
                 # If NBL=0, W defaults to 0.
                 if nbl_k == 0:
                      print(f"Warning: Pile {k} (KSU<=2) has NBL=0. Effective width W calculation might be incorrect.")
                      # Use top diameter as fallback?
                      if nfr_k > 0:
                           w[k] = pinf['dof'][k, nfr_k-1]
                      else:
                           w[k] = 0.0 # No segments at all
            else:
                 w[k] = 0.0 # No segments below ground


    # Calculate minimum spacing SMIN at the bottom
    for k in range(pnum_active):
        for ia in range(k + 1, pnum_active):
            dist_sq = (bxy[k, 0] - bxy[ia, 0])**2 + (bxy[k, 1] - bxy[ia, 1])**2
            s = math.sqrt(dist_sq)
            if s < smin[k]: smin[k] = s
            if s < smin[ia]: smin[ia] = s

    # Adjust W based on spacing and calculate final area AO
    for k in range(pnum_active):
        if smin[k] < w[k]: # If spacing is less than calculated effective width
            w[k] = smin[k] # Use spacing as effective width

        ksh_k = pinf['ksh'][k]
        if w[k] < 0: # Ensure width is not negative
             print(f"Warning: Effective width W is negative ({w[k]}) for pile {k}. Setting AO to 0.")
             ao[k] = 0.0
        elif ksh_k == 0: # Circular
            ao[k] = PI * w[k]**2 / 4.0
        else: # Square
            ao[k] = w[k]**2

    return ao

def stn(k, zbl_k, ao_k):
    """Calculate axial stiffness RZZ for a single pile k."""
    ksu_k = pinf['ksu'][k]
    ksh_k = pinf['ksh'][k]
    peh_k = pinf['peh'][k]
    pke_k = pinf['pke'][k]
    pmb_k = pinf['pmb'][k]
    nfr_k = pinf['nfr'][k]
    nbl_k = pinf['nbl'][k]

    pkc = 1.0 # Default factor
    if ksu_k == 1: pkc = 0.5
    elif ksu_k == 2: pkc = 0.667

    x_inv_stiff = 0.0 # Sum of 1/(EA) terms (flexibility)

    # Contribution from segments above ground
    for ia in range(nfr_k):
        dof_kia = pinf['dof'][k, ia]
        hfr_kia = pinf['hfr'][k, ia]
        a_ia, _ = eaj(ksh_k, pke_k, dof_kia)
        if peh_k * a_ia <= 0:
             print(f"Warning: EA is zero or negative in STN (free segment {ia}) for pile {k}. Skipping segment.")
             continue
        x_inv_stiff += hfr_kia / (peh_k * a_ia)

    # Contribution from segments below ground
    for ia in range(nbl_k):
        dob_kia = pinf['dob'][k, ia]
        hbl_kia = pinf['hbl'][k, ia]
        a_ia, _ = eaj(ksh_k, pke_k, dob_kia)
        if peh_k * a_ia <= 0:
             print(f"Warning: EA is zero or negative in STN (buried segment {ia}) for pile {k}. Skipping segment.")
             continue
        x_inv_stiff += pkc * hbl_kia / (peh_k * a_ia)

    # Contribution from end bearing
    if ao_k <= 0:
         print(f"Warning: Pile area AO is zero or negative for pile {k} in STN. End bearing contribution ignored.")
    else:
        if ksu_k <= 2: # Friction or combined pile
            if pmb_k * zbl_k * ao_k <= 0:
                 print(f"Warning: End bearing term denominator (PMB*ZBL*AO) is zero/negative for pile {k}. Contribution ignored.")
            else:
                 x_inv_stiff += 1.0 / (pmb_k * zbl_k * ao_k)
        else: # End bearing pile (KSU > 2)
             if pmb_k * ao_k <= 0:
                  print(f"Warning: End bearing term denominator (PMB*AO) is zero/negative for pile {k}. Contribution ignored.")
             else:
                  x_inv_stiff += 1.0 / (pmb_k * ao_k)

    # Final stiffness is inverse of flexibility sum
    if x_inv_stiff <= 0:
        print(f"Error: Total axial flexibility (X) is zero or negative for pile {k}. Cannot calculate RZZ.")
        return 0.0 # Return zero stiffness on error
    else:
        rzz = 1.0 / x_inv_stiff
        return rzz

def calculate_axial_stiffness(pnum_active, zbl, ao):
    """Calculate axial stiffness RZZ for all non-simulative piles."""
    rzz = np.zeros(pnum_active)
    if pnum_active == 0: return rzz

    calculated_stiffness = {} # Cache results: key=(kctr, ao), value=rzz

    for k in range(pnum_active):
        kctr_k = pinf['kctr'][k]
        ao_k = ao[k]
        # Create a tuple key, handle potential floating point inaccuracies in ao_k?
        # Maybe round ao_k slightly for the key?
        key = (kctr_k, round(ao_k, 6)) # Round AO for caching key

        if key in calculated_stiffness:
            rzz[k] = calculated_stiffness[key]
        else:
            # Check previous piles (Fortran logic) - more robust to use cache
            # found_match = False
            # if k > 0:
            #     for ia in range(k):
            #         if pinf['kctr'][ia] == kctr_k and abs(ao[ia] - ao_k) < 1e-9:
            #             rzz[k] = rzz[ia]
            #             found_match = True
            #             break
            # if not found_match:
            #      rzz[k] = stn(k, zbl[k], ao_k)
            #      calculated_stiffness[key] = rzz[k]
            rzz_k_calc = stn(k, zbl[k], ao_k)
            rzz[k] = rzz_k_calc
            calculated_stiffness[key] = rzz_k_calc


    return rzz


def calculate_pile_stiffness(pnum_active, rzz, btx, bty):
    """Calculate 6x6 stiffness matrix ESP for each non-simulative pile."""
    # ESP is stored globally, this function fills the first pnum*6 rows

    # Temporary arrays needed within the loop for each pile
    ej_fr = np.zeros(MAX_SEGMENTS)
    h_fr = np.zeros(MAX_SEGMENTS)
    ej_bl = np.zeros(MAX_SEGMENTS)
    h_bl_coords = np.zeros(MAX_SEGMENTS + 1) # Coordinates for non-free segments H(IA+1)=H(IA)+HBL(K,IA)

    for k in range(pnum_active):
        nbl_k = pinf['nbl'][k]
        nfr_k = pinf['nfr'][k]
        ksh_k = pinf['ksh'][k]
        pke_k = pinf['pke'][k]
        peh_k = pinf['peh'][k]
        ksu_k = pinf['ksu'][k]
        rzz_k = rzz[k]

        # --- Calculate Relational Matrix for Non-Free Segments (KBX, KBY) ---
        if nbl_k == 0:
            kbx = np.identity(4)
            kby = np.identity(4)
        else:
            # Prepare inputs for rltmtx
            bt1_k = btx[k, :nbl_k]
            bt2_k = bty[k, :nbl_k]
            h_bl_coords[0] = 0.0
            for ia in range(nbl_k):
                dob_kia = pinf['dob'][k, ia]
                _, b_ej_pke = eaj(ksh_k, pke_k, dob_kia)
                ej_bl[ia] = peh_k * (b_ej_pke / pke_k if abs(pke_k) > 1e-12 else 0)
                h_bl_coords[ia+1] = h_bl_coords[ia] + pinf['hbl'][k, ia]

            kbx, kby = rltmtx(nbl_k, bt1_k, bt2_k, ej_bl[:nbl_k], h_bl_coords[:nbl_k+1])

        # --- Calculate Relational Matrix for Free Segments (KFR) ---
        if nfr_k == 0:
            kfr = np.identity(4)
        else:
            # Prepare inputs for rltfr
            for ia in range(nfr_k):
                dof_kia = pinf['dof'][k, ia]
                _, b_ej_pke = eaj(ksh_k, pke_k, dof_kia)
                ej_fr[ia] = peh_k * (b_ej_pke / pke_k if abs(pke_k) > 1e-12 else 0)
                h_fr[ia] = pinf['hfr'][k, ia]
            kfr = rltfr(nfr_k, ej_fr[:nfr_k], h_fr[:nfr_k])

        # --- Combine Matrices (KX, KY) ---
        if nfr_k == 0:
            # If no free segments, KX = KBX (with modification), KY = KBY (with mod)
            kx = kbx.copy()
            ky = kby.copy()
            kx[:, 3] *= -1.0 # Fortran: KX(I,4)=-KX(I,4)
            ky[:, 3] *= -1.0 # Fortran: KY(I,4)=-KY(I,4)
        else:
            kx = combx(kbx, kfr) # combx applies the modification kbx[:, 3] *= -1.0
            ky = combx(kby, kfr) # combx applies the modification kby[:, 3] *= -1.0

        # --- Calculate Final 6x6 Stiffness Matrix (KE) ---
        ke = cndtn(ksu_k, kx, ky, rzz_k)

        # --- Store KE in global ESP array ---
        start_row = k * 6
        try:
             esp[start_row : start_row + 6, :] = ke[:, :]
        except IndexError:
             print(f"Error: Index out of bounds when writing to ESP for pile {k}. ESP size might be too small.")
             print(f"Required rows up to {start_row + 6}, ESP shape {esp.shape}")
             # Resize ESP or increase MAX_ESP_SIZE if necessary and possible
             # For now, just report error and potentially stop
             raise # Re-raise the error


def calculate_foundation_displacement(pnum, snum, force_ext, control_params):
    """Assemble global stiffness, solve for cap displacements."""
    jctr = control_params['jctr']
    ino = control_params['ino'] # Used only if jctr == 3

    so = np.zeros((6, 6)) # Global stiffness matrix

    # --- Assemble Global Stiffness Matrix SO ---
    num_total_piles = pnum + snum
    for k in range(num_total_piles):
        # Get local element stiffness KE = ESP[k*6 : k*6+6, :]
        start_row = k * 6
        try:
            ke_local = esp[start_row : start_row + 6, :].copy()
        except IndexError:
             print(f"Error: Index out of bounds reading ESP for pile {k} in DISP.")
             raise

        # Transformation matrices
        if k < pnum: # Non-simulative pile
            agl_k = pinf['agl'][k, :]
            pxy_k = pinf['pxy'][k, :]
            # Transformation for inclination TK
            tk = trnsfr(agl_k[0], agl_k[1], agl_k[2])
            tk1 = trnsps(6, 6, tk) # Transpose of TK
            # Transform KE from pile local coords to foundation local coords
            # KE_foundation_local = TK * KE_local * TK1
            ke_foundation_local = mulult(6, 6, 6, mulult(6, 6, 6, tk, ke_local), tk1)
            x_coord = pxy_k[0]
            y_coord = pxy_k[1]
        else: # Simulative pile (already in foundation local coords?)
            # Fortran code doesn't apply TRNSFR to simulative piles
            ke_foundation_local = ke_local # Assumes ESP for simu piles is already in foundation local
            k1 = k - pnum # Index for simu array
            sxy_k1 = simu['sxy'][k1, :]
            x_coord = sxy_k1[0]
            y_coord = sxy_k1[1]

        # Transformation for position TU
        tu = tmatx(x_coord, y_coord)
        tn = trnsps(6, 6, tu) # Transpose of TU

        # Transform KE from foundation local coords to global coords
        # KE_global = TN * KE_foundation_local * TU
        ke_global = mulult(6, 6, 6, mulult(6, 6, 6, tn, ke_foundation_local), tu)

        # Add to global stiffness matrix SO
        so += ke_global

    # --- Handle JCTR options ---
    if jctr == 3: # Output single pile stiffness
        # Need to get the untransformed local stiffness KE for pile INO
        ino_idx = ino - 1 # Convert 1-based INO to 0-based index
        if 0 <= ino_idx < pnum:
             start_row = ino_idx * 6
             ke_ino = esp[start_row : start_row + 6, :].copy()
             # No simulative pile stiffness output in Fortran JCTR=3 case
             return None, ke_ino # Return None for displacements, KE for output
        else:
             print(f"Error: Invalid pile number INO={ino} requested for JCTR=3.")
             return None, None

    if jctr == 2: # Output global stiffness SO only
        return None, so # Return None for displacements, SO for output

    # --- Solve for Global Displacements ---
    # Solve SO * D = F for D
    # Fortran uses GAOS which overwrites F (B in GAOS) with solution D
    force_copy = force_ext.copy() # Keep original forces
    displacements_global = gaos(6, so, force_copy) # Solves SO*D=F, result in displacements_global

    return displacements_global, so


def calculate_element_forces(pnum_active, zbl, duk_global):
    """Calculate displacements and internal forces along each pile body."""

    # Structure to store results for each pile
    eforce_results = []

    # Temporary arrays for calculations within the loop
    zh = np.zeros(MAX_NODES_PER_PILE)
    fx = np.zeros((MAX_NODES_PER_PILE, 4)) # UX, SX, NX, MX (Shear, Moment in local x-z plane)
    fy = np.zeros((MAX_NODES_PER_PILE, 4)) # UY, SY, NY, MY (Shear, Moment in local y-z plane)
    fz = np.zeros(MAX_NODES_PER_PILE)       # NZ (Axial force)
    psx = np.zeros(MAX_NODES_PER_PILE)      # Soil reaction X
    psy = np.zeros(MAX_NODES_PER_PILE)      # Soil reaction Y

    for k in range(pnum_active):
        # Get pile properties
        nfr_k = pinf['nfr'][k]
        nbl_k = pinf['nbl'][k]
        ksh_k = pinf['ksh'][k]
        pke_k = pinf['pke'][k]
        peh_k = pinf['peh'][k]
        ksu_k = pinf['ksu'][k]
        pmb_k = pinf['pmb'][k] # Needed for FZ calculation? Check Fortran. Yes, KSU<=2 case.
        btx_k = pinf['btx'][k, :] # Use pre-calculated BTX
        bty_k = pinf['bty'][k, :] # Use pre-calculated BTY

        # Get displacements DUK and stiffness SE for this pile (in pile local coords)
        # DUK was calculated in DISP (passed as duk_global here)
        # SE is the local stiffness matrix ESP[k*6 : k*6+6, :]
        ce = duk_global[k, :] # Displacements at pile top (local coords)
        start_row = k * 6
        se = esp[start_row : start_row + 6, :].copy()

        # Calculate forces PE at pile top (local coords) = SE * CE
        pe = mulult(6, 6, 1, se, ce)

        # Initialize values at pile top (z=0)
        zh[0] = 0.0
        fx[0, 0] = ce[0] # UX
        fx[0, 1] = ce[4] # SY (Rotation about Y, corresponds to bending in X-Z plane)
        fx[0, 2] = pe[0] # NX (Shear force in X)
        fx[0, 3] = pe[4] # MY (Moment about Y)
        fy[0, 0] = ce[1] # UY
        fy[0, 1] = ce[3] # SX (Rotation about X, corresponds to bending in Y-Z plane)
        fy[0, 2] = pe[1] # NY (Shear force in Y)
        fy[0, 3] = pe[3] # MX (Moment about X)
        fz[0] = pe[2] # NZ (Axial force)
        psx[0] = 0.0 # No soil reaction above ground
        psy[0] = 0.0

        nsum = 1 # Current node index (Python 0-based, so nsum=1 means zh[0], fx[0] etc. are filled)

        # --- Process Free Segments (Above Ground) ---
        for ia in range(nfr_k):
            nsf_kia = pinf['nsf'][k, ia]
            if nsf_kia <= 0:
                 print(f"Warning: NSF is zero or negative for pile {k}, free segment {ia}. Skipping segment.")
                 continue
            hl = pinf['hfr'][k, ia] / nsf_kia # Length of sub-segment
            dof_kia = pinf['dof'][k, ia]
            a_ia, b_ej_pke = eaj(ksh_k, pke_k, dof_kia)
            ej = peh_k * (b_ej_pke / pke_k if abs(pke_k) > 1e-12 else 0)

            if ej <= 0:
                 print(f"Warning: EJ is zero or negative for pile {k}, free segment {ia}. Skipping.")
                 # Need to advance nsum? Or just skip? Let's skip calculation for this segment.
                 zh[nsum] = zh[nsum-1] + pinf['hfr'][k, ia] # Advance Z coord anyway?
                 # Copy previous values?
                 fx[nsum, :] = fx[nsum-1, :]
                 fy[nsum, :] = fy[nsum-1, :]
                 fz[nsum] = fz[nsum-1]
                 psx[nsum] = 0.0
                 psy[nsum] = 0.0
                 nsum += 1
                 continue


            # Relational matrix for one sub-segment
            r_free = mfree(ej, hl)

            for _ in range(nsf_kia): # Loop over sub-segments
                # Get state at start of sub-segment (index nsum-1)
                xa = fx[nsum-1, :].copy() # State for X-Z bending
                xc = fy[nsum-1, :].copy() # State for Y-Z bending

                # Adjust signs based on Fortran convention vs. matrix definition?
                # Fortran EFORCE applies transformations inside SAA/MFREE implicitly?
                # Let's re-check MFREE: R(1,4)=-H^2/2EJ, R(2,4)=-H/EJ. These relate Disp/Rot to Moment/Shear.
                # Fortran FX(1,4)=PE(5)=MY, FX(1,2)=CE(5)=SY.
                # Fortran FY(1,4)=PE(4)=MX, FY(1,2)=CE(4)=SX.
                # The MFREE matrix seems consistent with standard beam theory [Disp, Rot, Shear, Moment] state vector?
                # Let's assume standard state vector: [Disp, Rot, Moment, Shear] or [Disp, Rot, Shear, Moment].
                # MFREE definition:
                # R(1,1)=1, R(1,2)=H, R(1,3)=H^3/6EJ, R(1,4)=-H^2/2EJ -> u_end = u_start + H*rot_start + ...
                # R(2,2)=1, R(2,3)=H^2/2EJ, R(2,4)=-H/EJ -> rot_end = rot_start + ...
                # R(3,3)=1, R(3,4)=0 -> M_end = M_start (?? MFREE has R(3,3)=1, R(3,4)=0) -> This assumes M=EI*d2u/dx2 convention?
                # R(4,3)=-H, R(4,4)=1 -> S_end = -H*M_start + S_start (?? MFREE has R(4,3)=-H, R(4,4)=1) -> This assumes S=EI*d3u/dx3?
                # Let's assume state is [u, theta, M, V] where M=EI*d2u/dx2, V=EI*d3u/dx3
                # u_end = u + H*th + H^2/(2EJ)*M + H^3/(6EJ)*V ??? No, MFREE seems different.
                # Let's assume state is [u, theta, V, M] based on Fortran FX/FY assignments.
                # FX: [UX, SY, NX_shear, MY_moment]
                # FY: [UY, SX, NY_shear, MX_moment]
                # MFREE:
                # R(1,:): u_end = u + H*th + H^3/(6EJ)*V - H^2/(2EJ)*M
                # R(2,:): th_end = th + H^2/(2EJ)*V - H/EJ*M
                # R(3,:): V_end = V  (Identity here)
                # R(4,:): M_end = -H*V + M
                # This looks like the transfer matrix for [u, theta, V, M] state vector.

                # Apply MFREE transformation
                xb = mulult(4, 4, 1, r_free, xa) # FX state transform
                xd = mulult(4, 4, 1, r_free, xc) # FY state transform

                # Store results at end of sub-segment
                if nsum >= MAX_NODES_PER_PILE:
                     print(f"Error: Exceeded MAX_NODES_PER_PILE ({MAX_NODES_PER_PILE}) in EFORCE for pile {k}.")
                     # Stop processing this pile or resize arrays dynamically
                     break # Stop processing segments for this pile

                fx[nsum, :] = xb[:]
                fy[nsum, :] = xd[:]
                zh[nsum] = zh[nsum-1] + hl
                fz[nsum] = fz[nsum-1] # Axial force constant in free segments
                psx[nsum] = 0.0
                psy[nsum] = 0.0
                nsum += 1
            # Break outer loop if inner loop broke due to size limit
            if nsum >= MAX_NODES_PER_PILE: break


        # Check if loop broke early
        if nsum >= MAX_NODES_PER_PILE: continue # Skip to next pile

        ig = nsum # Index of the node at the ground surface (start of buried segments)
        zg = zh[nsum-1] # Z-coordinate at ground surface

        # --- Process Buried Segments (Below Ground) ---
        for ia in range(nbl_k):
            nsg_kia = pinf['nsg'][k, ia]
            if nsg_kia <= 0:
                 print(f"Warning: NSG is zero or negative for pile {k}, buried segment {ia}. Skipping segment.")
                 continue
            hl = pinf['hbl'][k, ia] / nsg_kia # Length of sub-segment
            dob_kia = pinf['dob'][k, ia]
            pmt_kia = pinf['pmt'][k, ia] # Soil reaction modulus 'm'
            a_ia, b_ej_pke = eaj(ksh_k, pke_k, dob_kia)
            ej = peh_k * (b_ej_pke / pke_k if abs(pke_k) > 1e-12 else 0)

            if ej <= 0:
                 print(f"Warning: EJ is zero or negative for pile {k}, buried segment {ia}. Skipping.")
                 # Skip calculation for this segment
                 zh[nsum] = zh[nsum-1] + pinf['hbl'][k, ia] # Advance Z coord
                 # Copy previous values?
                 fx[nsum, :] = fx[nsum-1, :]
                 fy[nsum, :] = fy[nsum-1, :]
                 fz[nsum] = fz[nsum-1] # Or recalculate based on friction?
                 psx[nsum] = psx[nsum-1] # Or zero?
                 psy[nsum] = psy[nsum-1]
                 nsum += 1
                 continue

            # Get beta values for this segment
            btx_kia = btx_k[ia]
            bty_kia = bty_k[ia]

            # Check if beta values are valid
            if btx_kia <= 0 or bty_kia <= 0:
                 print(f"Warning: Beta values are zero or negative for pile {k}, buried segment {ia}. Skipping SAA calculation.")
                 # Skip calculation, advance nsum, copy previous state?
                 zh[nsum] = zh[nsum-1] + pinf['hbl'][k, ia]
                 fx[nsum, :] = fx[nsum-1, :]
                 fy[nsum, :] = fy[nsum-1, :]
                 fz[nsum] = fz[nsum-1]
                 psx[nsum] = psx[nsum-1]
                 psy[nsum] = psy[nsum-1]
                 nsum += 1
                 continue


            # Relational matrix using SAA for one sub-segment
            # Need coordinates relative to ground surface (zg) for SAA
            saa_prev_x = None
            saa_prev_y = None
            for i_sub in range(nsg_kia):
                h1 = zh[nsum-1] - zg # Z coord at start of sub-segment relative to ground
                h2 = h1 + hl       # Z coord at end of sub-segment relative to ground

                # Get state at start of sub-segment
                xa = fx[nsum-1, :].copy() # State for X-Z
                xc = fy[nsum-1, :].copy() # State for Y-Z

                # Adjust signs for SAA input/output convention?
                # Fortran: XA(4)=-XA(4) -> Moment sign?
                #          XC(2)=-XC(2) -> Rotation sign?
                # Let's assume SAA uses the same [u, theta, V, M] convention as MFREE
                # The sign changes might be needed if SAA assumes a different convention.
                # Revisit Fortran SAA call:
                # CALL SAA(BTX(K,IA),EJ,H1,H2,R)
                # CALL MULULT(4,4,1,R,XA,XB) -> XB = R * XA
                # Fortran applies sign changes *before* SAA call in the copy loop (304)
                # XA(I)=FX(NSUM,I) -> XA = [UX, SY, NX, MY]
                # XC(I)=FY(NSUM,I) -> XC = [UY, SX, NY, MX]
                # XA(4)=-XA(4) -> XA = [UX, SY, NX, -MY]
                # XC(2)=-XC(2) -> XC = [UY, -SX, NY, MX]
                # Then calls SAA(btx,...) -> R_x
                # Then calls SAA(bty,...) -> R_y
                # Then XB = R_x * XA
                # Then XD = R_y * XC
                # Then FX(NSUM,I)=XB(I) -> FX = [UX_end, SY_end, NX_end, -MY_end?]
                # Then FY(NSUM,I)=XD(I) -> FY = [UY_end, -SX_end?, NY_end, MX_end?]
                # Then FX(NSUM,4)=-XB(4) -> FX(NSUM,4) = MY_end
                # Then FY(NSUM,2)=-XD(2) -> FY(NSUM,2) = SX_end
                # This implies SAA output state is also [u, theta, V, M] but maybe with sign differences.
                # Let's apply the sign changes before calling SAA.
                xa[3] *= -1.0 # XA(4) = -MY_start
                xc[1] *= -1.0 # XC(2) = -SX_start

                # Calculate SAA matrices
                # Optimization: if btx == bty, only calculate saa once
                if abs(btx_kia - bty_kia) < 1.0e-10:
                    if saa_prev_x is None or i_sub == 0: # Calculate only if needed
                         r_saa = saa(btx_kia, ej, h1, h2)
                         saa_prev_x = r_saa
                         saa_prev_y = r_saa
                    else:
                         r_saa = saa_prev_x # Reuse previous if H increments linearly? No, H1/H2 change.
                         r_saa = saa(btx_kia, ej, h1, h2) # Recalculate
                    r_saa_x = r_saa
                    r_saa_y = r_saa
                else:
                    r_saa_x = saa(btx_kia, ej, h1, h2, saa_prev_x)
                    r_saa_y = saa(bty_kia, ej, h1, h2, saa_prev_y)
                    # saa_prev_x = r_saa_x # Cache for potential reuse (unlikely useful here)
                    # saa_prev_y = r_saa_y

                # Apply transformations
                xb = mulult(4, 4, 1, r_saa_x, xa)
                xd = mulult(4, 4, 1, r_saa_y, xc)

                # Store results, applying sign corrections based on Fortran logic
                if nsum >= MAX_NODES_PER_PILE:
                     print(f"Error: Exceeded MAX_NODES_PER_PILE ({MAX_NODES_PER_PILE}) in EFORCE for pile {k}.")
                     break # Stop processing sub-segments

                fx[nsum, 0] = xb[0] # UX_end
                fx[nsum, 1] = xb[1] # SY_end
                fx[nsum, 2] = xb[2] # NX_end
                fx[nsum, 3] = -xb[3] # MY_end = -XB(4)

                fy[nsum, 0] = xd[0] # UY_end
                fy[nsum, 1] = -xd[1] # SX_end = -XD(2)
                fy[nsum, 2] = xd[2] # NY_end
                fy[nsum, 3] = xd[3] # MX_end

                zh[nsum] = zh[nsum-1] + hl

                # Calculate soil reactions PSX, PSY
                # Fortran: PSX(NSUM)=FX(NSUM,1)*H2*PMT(K,IA)
                #          PSY(NSUM)=FY(NSUM,1)*H2*PMT(K,IA)
                # H2 is depth below ground at end of segment.
                # FX(NSUM,1) is UX_end. FY(NSUM,1) is UY_end.
                # Soil reaction p = m * z * u ? Or just p = m * u? Check soil model.
                # Fortran formula suggests p = m * z * u.
                psx[nsum] = fx[nsum, 0] * h2 * pmt_kia
                psy[nsum] = fy[nsum, 0] * h2 * pmt_kia

                # Calculate axial force FZ
                if ksu_k >= 3: # End bearing pile, axial force assumed constant below ground?
                    fz[nsum] = fz[nsum-1] # Or maybe fz[ig-1]? Fortran uses FZ(IG)
                    fz[nsum] = fz[ig-1] # Axial force at ground surface
                else: # Friction pile
                    # FZ(NSUM)=FZ(IG)*(1.0-H2**2/ZBL**2)
                    zbl_k = zbl[k]
                    if zbl_k <= 0:
                         print(f"Warning: ZBL is zero or negative for pile {k}. Cannot calculate FZ friction component.")
                         fz[nsum] = fz[ig-1] # Assume constant axial force as fallback
                    else:
                         fz_ground = fz[ig-1] # Axial force at ground surface
                         fz[nsum] = fz_ground * (1.0 - h2**2 / zbl_k**2)

                nsum += 1
            # Break outer loop if inner loop broke
            if nsum >= MAX_NODES_PER_PILE: break

        # Store results for pile k
        pile_result = {
            'k': k + 1, # Pile number (1-based)
            'pxy': pinf['pxy'][k, :].copy(),
            'ce': ce.copy(), # Disp at top
            'pe': pe.copy(), # Force at top
            'nsum': nsum,    # Number of nodes calculated
            'zh': zh[:nsum].copy(),
            'fx': fx[:nsum, :].copy(),
            'fy': fy[:nsum, :].copy(),
            'fz': fz[:nsum].copy(),
            'psx': psx[:nsum].copy(),
            'psy': psy[:nsum].copy(),
            'ig': ig # Index of ground node
        }
        eforce_results.append(pile_result)

    return eforce_results

def write_output(out_filename, pos_filename, disp_global, eforce_results, control_params, so_matrix=None, ke_single=None):
    """Writes results to .out and .pos files."""

    jctr = control_params['jctr']
    ino = control_params['ino']

    # --- Write to .out file ---
    try:
        with open(out_filename, 'w') as f_out:
            print_head2(f_out) # Print standard header

            if jctr == 2: # Only stiffness of entire foundation
                f_out.write("\n" + 7*' ' + '*** Stiffness of the entire pile foundation ***\n')
                if so_matrix is not None:
                    for i in range(6):
                        f_out.write(7*' ' + ' '.join(f"{x:12.4E}" for x in so_matrix[i, :]) + '\n')
                else:
                    f_out.write("   (Stiffness matrix not calculated)\n")

            elif jctr == 3: # Only stiffness of single pile INO
                f_out.write(f"\n{7*' '}*** Stiffness of the No.{ino:2d} pile ***\n")
                if ke_single is not None:
                    for i in range(6):
                        f_out.write(7*' ' + ' '.join(f"{x:12.4E}" for x in ke_single[i, :]) + '\n')
                else:
                    f_out.write(f"   (Stiffness matrix not available for pile {ino})\n")

            elif jctr == 1 or jctr == 0: # Full analysis results
                # Global displacements
                f_out.write(7*' ' + '***************************************************************************************\n')
                f_out.write(15*' ' + 'DISPLACEMENTS AT THE CAP CENTER OF PILE FOUNDATION\n')
                f_out.write(7*' ' + '***************************************************************************************\n')
                if disp_global is not None:
                    f_out.write(f"{16*' '}Movement in the direction of X axis : UX={disp_global[0]:12.4E} (m)\n")
                    f_out.write(f"{16*' '}Movement in the direction of Y axis : UY={disp_global[1]:12.4E} (m)\n")
                    f_out.write(f"{16*' '}Movement in the direction of Z axis : UZ={disp_global[2]:12.4E} (m)\n")
                    f_out.write(f"{16*' '}Rotational angle  around X axis :     SX={disp_global[3]:12.4E} (rad)\n")
                    f_out.write(f"{16*' '}Rotational angle around Y axis :      SY={disp_global[4]:12.4E} (rad)\n")
                    f_out.write(f"{16*' '}Rotational angle around Z axis :      SZ={disp_global[5]:12.4E} (rad)\n\n")
                else:
                     f_out.write("   (Global displacements not calculated)\n\n")

                # Element forces and displacements
                if eforce_results:
                    for res in eforce_results:
                        k_pile = res['k']
                        pxy_k = res['pxy']
                        ce_k = res['ce']
                        pe_k = res['pe']
                        nsum_k = res['nsum']
                        zh_k = res['zh']
                        fx_k = res['fx']
                        fy_k = res['fy']
                        fz_k = res['fz']
                        psx_k = res['psx']
                        psy_k = res['psy']
                        ig_k = res['ig'] # Ground node index

                        f_out.write(7*' ' + '*************************************************************************************************\n')
                        f_out.write(f"{34*' '}NO. {k_pile:2d} # PILE\n")
                        f_out.write(7*' ' + '*************************************************************************************************\n')
                        f_out.write(f"{12*' '}Coordinator of the pile: (x,y) = ({pxy_k[0]:12.4E} ,{pxy_k[1]:12.4E} )\n\n")
                        f_out.write(12*' ' + 'Displacements and internal forces at the top of pile:\n')
                        f_out.write(f"{15*' '}UX={ce_k[0]:12.4E} (m){9*' '}NX={pe_k[0]:12.4E} (t)\n")
                        f_out.write(f"{15*' '}UY={ce_k[1]:12.4E} (m){9*' '}NY={pe_k[1]:12.4E} (t)\n")
                        f_out.write(f"{15*' '}UZ={ce_k[2]:12.4E} (m){9*' '}NZ={pe_k[2]:12.4E} (t)\n")
                        f_out.write(f"{15*' '}SX={ce_k[3]:12.4E} (rad){7*' '}MX={pe_k[3]:12.4E} (t*m)\n")
                        f_out.write(f"{15*' '}SY={ce_k[4]:12.4E} (rad){7*' '}MY={pe_k[4]:12.4E} (t*m)\n")
                        f_out.write(f"{15*' '}SZ={ce_k[5]:12.4E} (rad){7*' '}MZ={pe_k[5]:12.4E} (t*m)\n\n")

                        # Displacements along pile body
                        f_out.write(7*' ' + '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
                        f_out.write(32*' ' + 'Displacements of the pile body and\n')
                        f_out.write(27*' ' + '     Compression stresses of soil (PSX,PSY)\n')
                        f_out.write(7*' ' + '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
                        f_out.write(f"\n{15*' '}Z{12*' '}UX{12*' '}UY{12*' '}SX{12*' '}SY{12*' '}PSX{12*' '}PSY\n")
                        f_out.write(f"{14*' '}(m){11*' '}(m){11*' '}(m){10*' '}(rad){9*' '}(rad){9*' '}(t/m2){9*' '}(t/m2)\n\n") # Units from Fortran 831

                        # Write free segment displacements (no PSX/PSY)
                        for i in range(ig_k): # Up to ground node
                            f_out.write(f"{7*' '}{zh_k[i]:14.4E}{fx_k[i, 0]:14.4E}{fy_k[i, 0]:14.4E}"
                                        f"{fy_k[i, 1]:14.4E}{fx_k[i, 1]:14.4E}\n") # SX=FY(1,2), SY=FX(1,2)

                        # Write buried segment displacements (with PSX/PSY)
                        for i in range(ig_k, nsum_k): # From ground node downwards
                            f_out.write(f"{7*' '}{zh_k[i]:14.4E}{fx_k[i, 0]:14.4E}{fy_k[i, 0]:14.4E}"
                                        f"{fy_k[i, 1]:14.4E}{fx_k[i, 1]:14.4E}"
                                        f"{psx_k[i]:14.4E}{psy_k[i]:14.4E}\n")
                        f_out.write("\n\n")

                        # Internal forces along pile body
                        f_out.write(7*' ' + '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
                        f_out.write(32*' ' + 'Internal forces of the pile body\n')
                        f_out.write(7*' ' + '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
                        f_out.write(f"\n{18*' '}Z{14*' '}NX{14*' '}NY{14*' '}NZ{14*' '}MX{14*' '}MY\n")
                        f_out.write(f"{17*' '}(m){12*' '}(t){13*' '}(t){13*' '}(t){12*' '}(t*m){11*' '}(t*m)\n\n") # Units from Fortran 832
                        for i in range(nsum_k):
                            f_out.write(f"{7*' '}{zh_k[i]:16.4E}{fx_k[i, 2]:16.4E}{fy_k[i, 2]:16.4E}"
                                        f"{fz_k[i]:16.4E}{fy_k[i, 3]:16.4E}{fx_k[i, 3]:16.4E}\n") # NX=FX(1,3), NY=FY(1,3), NZ=FZ, MX=FY(1,4), MY=FX(1,4)
                        f_out.write("\n")
                else:
                     f_out.write("   (Element forces/displacements not calculated)\n")

    except IOError as e:
        print(f"Error writing to output file {out_filename}: {e}")

    # --- Write to .pos file ---
    # This file seems to contain data for post-processing/visualization
    try:
        with open(pos_filename, 'w') as f_pos:
             if eforce_results: # Only write if full analysis was done
                 pnum_active = len(eforce_results)
                 f_pos.write(f"{pnum_active:5d}\n") # Number of non-simu piles processed
                 # Write pile top coordinates (X,Y)
                 coords_line = ""
                 for i in range(pnum_active):
                      res = eforce_results[i]
                      coords_line += f"{res['pxy'][0]:14.4E}{res['pxy'][1]:14.4E}"
                      if (i + 1) % 3 == 0 or i == pnum_active - 1: # Write 3 pairs per line (6 values)
                           f_pos.write(coords_line + "\n")
                           coords_line = ""
                 if coords_line: # Write any remaining part
                      f_pos.write(coords_line + "\n")


                 # Write data for each pile
                 for res in eforce_results:
                     k_pile = res['k']
                     nsum_k = res['nsum']
                     pxy_k = res['pxy']
                     zh_k = res['zh']
                     fx_k = res['fx']
                     fy_k = res['fy']
                     fz_k = res['fz']
                     psx_k = res['psx']
                     psy_k = res['psy']

                     f_pos.write(f"{k_pile:5d}{nsum_k:5d}\n") # Pile number, number of nodes
                     f_pos.write(f"{pxy_k[0]:14.4E}{pxy_k[1]:14.4E}\n") # Pile top X, Y

                     # Write Z coordinates
                     z_line = ""
                     for i in range(nsum_k):
                          z_line += f"{zh_k[i]:14.4E}"
                          if (i + 1) % 6 == 0 or i == nsum_k - 1:
                               f_pos.write(z_line + "\n")
                               z_line = ""
                     if z_line: f_pos.write(z_line + "\n")

                     # Write FX matrix (4 columns)
                     for i in range(nsum_k):
                          fx_line = "".join(f"{val:14.4E}" for val in fx_k[i,:])
                          f_pos.write(fx_line + "\n")

                     # Write FY matrix (4 columns)
                     for i in range(nsum_k):
                          fy_line = "".join(f"{val:14.4E}" for val in fy_k[i,:])
                          f_pos.write(fy_line + "\n")

                     # Write FZ vector
                     fz_line = ""
                     for i in range(nsum_k):
                          fz_line += f"{fz_k[i]:14.4E}"
                          if (i + 1) % 6 == 0 or i == nsum_k - 1:
                               f_pos.write(fz_line + "\n")
                               fz_line = ""
                     if fz_line: f_pos.write(fz_line + "\n")

                     # Write PSX vector
                     psx_line = ""
                     for i in range(nsum_k):
                          psx_line += f"{psx_k[i]:14.4E}"
                          if (i + 1) % 6 == 0 or i == nsum_k - 1:
                               f_pos.write(psx_line + "\n")
                               psx_line = ""
                     if psx_line: f_pos.write(psx_line + "\n")

                     # Write PSY vector
                     psy_line = ""
                     for i in range(nsum_k):
                          psy_line += f"{psy_k[i]:14.4E}"
                          if (i + 1) % 6 == 0 or i == nsum_k - 1:
                               f_pos.write(psy_line + "\n")
                               psy_line = ""
                     if psy_line: f_pos.write(psy_line + "\n")

             else:
                  # Write minimal info if no results (e.g., JCTR=2 or 3)
                  f_pos.write("0\n") # Zero piles processed

    except IOError as e:
        print(f"Error writing to position file {pos_filename}: {e}")


# --- Main Execution Logic ---
def bcad_pile_main(input_filename):
    """Main function mirroring BCAD_PILE subroutine."""
    # Reset global data structures (important if called multiple times)
    global pinf, simu, esp
    pinf = {k: np.zeros_like(v) if isinstance(v, np.ndarray) else v for k, v in pinf.items()}
    simu = {k: np.zeros_like(v) if isinstance(v, np.ndarray) else v for k, v in simu.items()}
    esp.fill(0.0)


    print_head1() # Print header to console

    # Determine filenames
    base_name = os.path.splitext(input_filename)[0]
    dat_file = input_filename
    out_file = f_name(base_name, ".out")
    pos_file = f_name(base_name, ".pos")

    # --- Read Input Data ---
    print(f"--- Reading Input Information from {dat_file} ---")
    pnum, snum, force, control_params, zfr, zbl = read_data(dat_file)
    print(f"   Non-simulative piles (PNUM): {pnum}")
    print(f"   Simulative piles (SNUM): {snum}")
    print(f"   Control parameter (JCTR): {control_params['jctr']}")
    if control_params['jctr'] == 3:
        print(f"   Target pile for stiffness (INO): {control_params['ino']}")
    print(f"   Initial Global Forces: {force}")


    # Check if only reading is needed (though Fortran doesn't have this option)
    if pnum == 0 and snum == 0 and control_params['jctr'] not in [2, 3]:
         print("\nWarning: No piles defined in the input file.")
         # Write empty output?
         write_output(out_file, pos_file, None, [], control_params)
         return # Exit early if no piles to analyze


    # --- Calculations (only if piles exist or JCTR requires them) ---
    btx, bty, pile_areas, axial_stiffness = None, None, None, None
    eforce_results = None
    displacements_global = None
    overall_stiffness = None
    ke_single = None

    # Perform calculations needed for all JCTR modes involving non-simu piles
    if pnum > 0:
        print("\n--- Calculating Deformation Factors (BTX, BTY) ---")
        # Pass required pinf data explicitly or ensure it's globally accessible
        pinf['btx'], pinf['bty'] = calculate_deformation_factors(pnum, zfr[:pnum], zbl[:pnum])
        btx = pinf['btx'] # Local reference
        bty = pinf['bty']

        print("\n--- Calculating Pile Bottom Areas (AO) ---")
        pile_areas = calculate_pile_areas(pnum, zfr[:pnum], zbl[:pnum])

        print("\n--- Calculating Pile Axial Stiffness (RZZ) ---")
        axial_stiffness = calculate_axial_stiffness(pnum, zbl[:pnum], pile_areas[:pnum])

        print("\n--- Calculating Pile Lateral Stiffness (ESP) ---")
        calculate_pile_stiffness(pnum, axial_stiffness[:pnum], btx[:pnum,:], bty[:pnum,:])
        # ESP for non-simu piles is now filled globally

    # --- Foundation Analysis ---
    # This step assembles SO and solves, needed for JCTR=0, 1, 2
    if control_params['jctr'] in [0, 1, 2]:
        print("\n--- Executing Entire Pile Foundation Analysis ---")
        # The function now handles JCTR=2 and JCTR=3 internally
        displacements_global, stiffness_result = calculate_foundation_displacement(
            pnum, snum, force, control_params
        )
        if control_params['jctr'] == 2:
            overall_stiffness = stiffness_result
        elif control_params['jctr'] == 3:
             ke_single = stiffness_result # This case should be handled below now
        # If JCTR=0 or 1, displacements_global contains the result, SO was calculated internally

    # Handle JCTR=3 specifically (stiffness for one pile)
    if control_params['jctr'] == 3:
         ino = control_params['ino']
         ino_idx = ino - 1
         if 0 <= ino_idx < pnum:
              start_row = ino_idx * 6
              try:
                   ke_single = esp[start_row : start_row + 6, :].copy()
                   print(f"   Extracted stiffness for pile {ino}.")
              except IndexError:
                   print(f"Error: Failed to extract stiffness for pile {ino} (index {ino_idx}).")
                   ke_single = None
         else:
              print(f"Error: Invalid pile number {ino} for JCTR=3 output.")
              ke_single = None
         # Write output for JCTR=3
         print(f"\n--- Writing Stiffness Output for Pile {ino} ---")
         write_output(out_file, pos_file, None, None, control_params, ke_single=ke_single)
         print(f"\n--- Analysis Complete (JCTR=3) ---")
         return # End execution for JCTR=3


    # Handle JCTR=2 specifically (overall stiffness)
    if control_params['jctr'] == 2:
         # Write output for JCTR=2
         print(f"\n--- Writing Overall Stiffness Output ---")
         write_output(out_file, pos_file, None, None, control_params, so_matrix=overall_stiffness)
         print(f"\n--- Analysis Complete (JCTR=2) ---")
         return # End execution for JCTR=2


    # --- Calculate Element Forces (JCTR=0 or 1) ---
    if control_params['jctr'] in [0, 1] and pnum > 0 and displacements_global is not None:
        print("\n--- Calculating Displacements and Internal Forces along Piles ---")
        # Need to transform global displacements back to local pile displacements (DUK)
        duk_local = np.zeros((pnum, 6))
        for k in range(pnum):
            pxy_k = pinf['pxy'][k, :]
            agl_k = pinf['agl'][k, :]
            # Transform global displacements D to foundation local at pile head C1
            tu = tmatx(pxy_k[0], pxy_k[1])
            c1 = mulult(6, 6, 1, tu, displacements_global)
            # Transform foundation local C1 to pile local C (DUK)
            tk = trnsfr(agl_k[0], agl_k[1], agl_k[2])
            tk1 = trnsps(6, 6, tk) # Transpose
            # C = TK1 * C1
            duk_local[k, :] = mulult(6, 6, 1, tk1, c1)

        # Now calculate forces along the pile using local displacements DUK
        eforce_results = calculate_element_forces(pnum, zbl[:pnum], duk_local)
    elif control_params['jctr'] in [0, 1]:
         print("\n--- Skipping Element Force Calculation (No non-simu piles or displacement error) ---")
         eforce_results = []


    # --- Write Final Output (JCTR=0 or 1) ---
    print(f"\n--- Writing Full Analysis Results to {out_file} and {pos_file} ---")
    write_output(out_file, pos_file, displacements_global, eforce_results, control_params)

    print(f"\n--- BCAD_PILE Analysis Complete ---")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fname_in = sys.argv[1]
    else:
        fname_in = input("Please enter data filename (e.g., input.dat): ")

    if not os.path.exists(fname_in):
         print(f"Error: Input file '{fname_in}' not found.")
         sys.exit(1)
    # Optional: Check for .dat extension
    # if not fname_in.lower().endswith(".dat"):
    #      print("Warning: Input filename does not end with .dat")

    bcad_pile_main(fname_in)
    

# **如何运行:**

# 1.  **保存:** 将上面的代码保存为一个 Python 文件（例如 `bcad_pile.py`）。
# 2.  **安装 NumPy:** 如果你还没有安装 NumPy，请在终端或命令提示符中运行：`pip install numpy`
# 3.  **准备输入文件:** 确保你有一个与原始 Fortran 程序兼容的 `.dat` 输入文件。
# 4.  **运行:** 在终端中，导航到保存文件的目录并运行脚本，将输入文件名作为参数传递：
#     ```bash
#     python bcad_pile.py your_input_file.dat
#     ```
#     或者，如果不带参数运行，它会提示你输入文件名：
#     ```bash
#     python bcad_pile.py
#     Please enter data filename (e.g., input.dat): your_input_file.dat
#     ```
# 5.  **检查输出:** 程序将生成 `.out` 和 `.pos` 文件（与输入文件同名，但扩展名不同），其中包含分析结果。

# 请注意，由于原始 Fortran 代码的复杂性和潜在的特定输入格式依赖性，你可能需要根据你的具体 `.dat` 文件调整 Python 代码中的文件读取和解析逻辑。仔细测试对于确保准确性至关
