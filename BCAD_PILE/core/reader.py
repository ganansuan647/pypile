"""
Data reader module for the BCAD_PILE package.

This module provides functions for reading input data files in the format
expected by the original Fortran code.
"""

import numpy as np
from ..utils.matrix import tmatx, trnsps, mulult


def read_input_file(filename, pile_data, sim_pile_data, element_stiffness):
    """
    Read pile foundation data from an input file.
    Equivalent to the R_DATA subroutine and related functions in the original Fortran code.

    Args:
        filename: Path to input file
        pile_data: PileData object
        sim_pile_data: SimulativePileData object
        element_stiffness: ElementStiffnessData object

    Returns:
        Tuple of (jctr, ino, pnum, snum, force, zfr, zbl)
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    line_index = 0

    # Read control block
    line_index += 1  # Skip title line
    jctr = int(lines[line_index].strip())
    line_index += 1

    force = np.zeros(6)
    ino = 0

    if jctr == 1:
        nact = int(lines[line_index].strip())
        line_index += 1

        axy = np.zeros((nact, 2))
        act = np.zeros((nact, 6))

        for i in range(nact):
            coords = list(map(float, lines[line_index].strip().split()))
            axy[i, 0] = coords[0]
            axy[i, 1] = coords[1]
            line_index += 1

            values = list(map(float, lines[line_index].strip().split()))
            act[i, :] = values
            line_index += 1

        # Calculate combined forces
        force = init6(nact, axy, act)

    if jctr == 2:
        pass  # Nothing special to do for mode 2

    if jctr == 3:
        ino = int(lines[line_index].strip())
        line_index += 1

    line_index += 1  # Skip TAG line

    # Read arrange block
    line_index += 1  # Skip TITLE line

    pnum_snum = list(map(int, lines[line_index].strip().split()))
    pnum = pnum_snum[0]
    snum = pnum_snum[1]
    line_index += 1

    # Initialize arrays for lengths
    zfr = np.zeros(pnum)
    zbl = np.zeros(pnum)

    # Read pile coordinates
    for k in range(pnum):
        coords = list(map(float, lines[line_index].strip().split()))
        pile_data.pxy[k, 0] = coords[0]
        pile_data.pxy[k, 1] = coords[1]
        line_index += 1

    if snum > 0:
        for k in range(snum):
            coords = list(map(float, lines[line_index].strip().split()))
            sim_pile_data.sxy[k, 0] = coords[0]
            sim_pile_data.sxy[k, 1] = coords[1]
            line_index += 1

    line_index += 1  # Skip TAG line

    # Read NO_SIMU block
    line_index += 1  # Skip TITLE line

    kctr_values = list(map(int, lines[line_index].strip().split()))
    for k in range(pnum):
        pile_data.kctr[k] = kctr_values[k]
    line_index += 1

    idf = init1(pile_data.kctr[:pnum])

    # Read <0> segment
    stag = lines[line_index].strip()
    line_index += 1

    if stag != "<0>":
        print("Error: <0> tag expected")
        return None

    init2(0, pnum, pile_data, lines, line_index)
    line_index += 3 + pile_data.nfr[0] + pile_data.nbl[0]  # Move past the segment data

    # Read other segments
    for _ in range(idf - 1):
        line = lines[line_index].strip()
        im = int(line[1:3])
        line_index += 1

        if im > 0:
            init2(im, pnum, pile_data, lines, line_index)
            line_index += (
                3 + pile_data.nfr[0] + pile_data.nbl[0]
            )  # Move past the segment data
        elif im < 0:
            jj = int(lines[line_index].strip())
            line_index += 1

            sig = []
            jnew = np.zeros(jj, dtype=int)
            vnew = np.zeros(jj)

            for ia in range(jj):
                line_parts = lines[line_index].strip().split()
                sig.append(line_parts[0])
                jnew[ia] = int(line_parts[1])
                vnew[ia] = float(line_parts[2])
                line_index += 1

            init4(im, jj, pnum, sig, jnew, vnew, pile_data)

    line_index += 1  # Skip TAG line

    # Read SIMUPILE block
    line_index += 1  # Skip TITLE line

    if snum > 0:
        ksctr_values = list(map(int, lines[line_index].strip().split()))
        for ks in range(snum):
            sim_pile_data.ksctr[ks] = ksctr_values[ks]
        line_index += 1

        idf = init1(sim_pile_data.ksctr[:snum])
        is_val = pnum * 6

        for _ in range(idf):
            line = lines[line_index].strip()
            im = int(line[1:3])
            line_index += 1

            init5(im, is_val, snum, sim_pile_data, element_stiffness, lines, line_index)

            if im < 0:
                line_index += 1  # Single line of 6 values
            else:
                line_index += 6  # Six lines of 6 values each

    # Calculate lengths of piles
    for k in range(pnum):
        zfr[k] = np.sum(pile_data.hfr[k, : pile_data.nfr[k]])
        zbl[k] = np.sum(pile_data.hbl[k, : pile_data.nbl[k]])

    return jctr, ino, pnum, snum, force, zfr, zbl


def init1(kctr):
    """
    Calculate the number of different control types.
    Equivalent to INIT1 in the original Fortran code.

    Args:
        kctr: Array of control values

    Returns:
        Number of different control types
    """
    pnum = len(kctr)
    idf = 1

    for k in range(1, pnum):
        found = False
        for ki in range(k):
            if kctr[k] == kctr[ki]:
                found = True
                break
        if not found:
            idf += 1

    return idf


def init2(im, pnum, pile_data, lines, line_index):
    """
    Read segment information for piles.
    Equivalent to INIT2 in the original Fortran code.

    Args:
        im: Segment index
        pnum: Number of piles
        pile_data: PileData object
        lines: List of file lines
        line_index: Current line index

    Returns:
        Updated line index
    """
    # Read shape, support, and direction cosines
    parts = lines[line_index].strip().split()
    ksh1 = int(parts[0])
    ksu1 = int(parts[1])
    agl1 = [float(parts[2]), float(parts[3]), float(parts[4])]
    line_index += 1

    # Read free segment data
    parts = lines[line_index].strip().split()
    nfr1 = int(parts[0])
    hfr1 = np.zeros(nfr1)
    dof1 = np.zeros(nfr1)
    nsf1 = np.zeros(nfr1, dtype=int)

    for i in range(nfr1):
        idx = 1 + i * 3
        hfr1[i] = float(parts[idx])
        dof1[i] = float(parts[idx + 1])
        nsf1[i] = int(parts[idx + 2])

    line_index += 1

    # Read buried segment data - first get the number of layers
    parts = lines[line_index].strip().split()
    nbl1 = int(parts[0])
    hbl1 = np.zeros(nbl1)
    dob1 = np.zeros(nbl1)
    pmt1 = np.zeros(nbl1)
    pfi1 = np.zeros(nbl1)
    nsg1 = np.zeros(nbl1, dtype=int)

    # 支持三种格式:
    # 1. 单行格式（传统格式）: 13 13.4 2.0 5000 18.0 14 7.1 2.0 3000 13.7 8 ...
    # 2. 多行格式（每层一行）: 13
    #                        13.4 2.0 5000 18.0 14
    #                        7.1 2.0 3000 13.7 8
    #                        ...
    # 3. 折中格式（第一层与层数在同一行）: 13 13.4 2.0 5000 18.0 14
    #                                   7.1 2.0 3000 13.7 8
    #                                   ...

    # 情况1：如果只有一个数字（土层数量），那么接下来的每一行是一个土层的数据（多行格式）
    if len(parts) == 1:
        for i in range(nbl1):
            line_index += 1  # 移动到下一行
            if line_index >= len(lines):
                raise ValueError(
                    f"Unexpected end of file while reading buried segment {i + 1}"
                )

            layer_parts = lines[line_index].strip().split()
            if len(layer_parts) < 5:
                raise ValueError(
                    f"Not enough parameters for buried segment {i + 1}: {layer_parts}"
                )

            hbl1[i] = float(layer_parts[0])
            dob1[i] = float(layer_parts[1])
            pmt1[i] = float(layer_parts[2])
            pfi1[i] = float(layer_parts[3])
            nsg1[i] = int(layer_parts[4])

    # 情况2：如果有足够的参数表示所有土层（单行格式）
    elif len(parts) >= 1 + nbl1 * 5:
        # 原有的单行格式处理
        for i in range(nbl1):
            idx = 1 + i * 5
            if idx + 4 >= len(parts):
                raise ValueError(
                    f"Not enough parameters for buried segment {i + 1}: {parts}"
                )

            hbl1[i] = float(parts[idx])
            dob1[i] = float(parts[idx + 1])
            pmt1[i] = float(parts[idx + 2])
            pfi1[i] = float(parts[idx + 3])
            nsg1[i] = int(parts[idx + 4])

    # 情况3：如果有6个参数（层数+第一层5个参数），那么是折中格式
    elif len(parts) == 6:
        # 处理第一层（与层数在同一行）
        hbl1[0] = float(parts[1])
        dob1[0] = float(parts[2])
        pmt1[0] = float(parts[3])
        pfi1[0] = float(parts[4])
        nsg1[0] = int(parts[5])

        # 处理剩余的层（每层一行）
        for i in range(1, nbl1):
            line_index += 1  # 移动到下一行
            if line_index >= len(lines):
                raise ValueError(
                    f"Unexpected end of file while reading buried segment {i + 1}"
                )

            layer_parts = lines[line_index].strip().split()
            if len(layer_parts) < 5:
                raise ValueError(
                    f"Not enough parameters for buried segment {i + 1}: {layer_parts}"
                )

            hbl1[i] = float(layer_parts[0])
            dob1[i] = float(layer_parts[1])
            pmt1[i] = float(layer_parts[2])
            pfi1[i] = float(layer_parts[3])
            nsg1[i] = int(layer_parts[4])

    else:
        raise ValueError(f"Invalid format for buried segment data: {parts}")

    line_index += 1

    # Read material properties
    parts = lines[line_index].strip().split()
    pmb1 = float(parts[0])
    peh1 = float(parts[1])
    pke1 = float(parts[2])
    line_index += 1

    # Assign to piles with matching control value
    for k in range(pnum):
        if init3(im, pile_data.kctr[k]):
            pile_data.ksh[k] = ksh1
            pile_data.ksu[k] = ksu1
            pile_data.agl[k, :] = agl1

            pile_data.nfr[k] = nfr1
            pile_data.hfr[k, :nfr1] = hfr1
            pile_data.dof[k, :nfr1] = dof1
            pile_data.nsf[k, :nfr1] = nsf1

            pile_data.nbl[k] = nbl1
            pile_data.hbl[k, :nbl1] = hbl1
            pile_data.dob[k, :nbl1] = dob1
            pile_data.pmt[k, :nbl1] = pmt1
            pile_data.pfi[k, :nbl1] = pfi1
            pile_data.nsg[k, :nbl1] = nsg1

            pile_data.pmb[k] = pmb1
            pile_data.peh[k] = peh1
            pile_data.pke[k] = pke1

    return line_index


def init3(im, k):
    """
    Test if a pile's control value matches the segment index.
    Equivalent to INIT3 in the original Fortran code.

    Args:
        im: Segment index
        k: Pile control value

    Returns:
        True if values match, False otherwise
    """
    if im == 0 and k <= 0:
        return True
    if im >= 0 and k == im:
        return True
    return False


def init4(im, jj, pnum, sig, jnew, vnew, pile_data):
    """
    Modify pile parameters based on segment specifications.
    Equivalent to INIT4 in the original Fortran code.

    Args:
        im: Segment index (negative)
        jj: Number of parameters to modify
        pnum: Number of piles
        sig: List of parameter names
        jnew: List of parameter indices
        vnew: List of parameter values
        pile_data: PileData object
    """
    # Find piles with matching control value
    nim = []
    for k in range(pnum):
        if pile_data.kctr[k] == im:
            nim.append(k)

    if not nim:
        print(f"Error: <{im}>")
        return

    # Modify parameters for matching piles
    for ia in range(jj):
        param_name = sig[ia]
        index = jnew[ia]
        value = vnew[ia]

        if param_name == "KSH=":
            for k in nim:
                pile_data.ksh[k] = int(value)
        elif param_name == "KSU=":
            for k in nim:
                pile_data.ksu[k] = int(value)
        elif param_name == "AGL=":
            for k in nim:
                pile_data.agl[k, index - 1] = value
        elif param_name == "NFR=":
            for k in nim:
                pile_data.nfr[k] = int(value)
        elif param_name == "HFR=":
            for k in nim:
                pile_data.hfr[k, index - 1] = value
        elif param_name == "DOF=":
            for k in nim:
                pile_data.dof[k, index - 1] = value
        elif param_name == "NSF=":
            for k in nim:
                pile_data.nsf[k, index - 1] = int(value)
        elif param_name == "NBL=":
            for k in nim:
                pile_data.nbl[k] = int(value)
        elif param_name == "HBL=":
            for k in nim:
                pile_data.hbl[k, index - 1] = value
        elif param_name == "DOB=":
            for k in nim:
                pile_data.dob[k, index - 1] = value
        elif param_name == "PMT=":
            for k in nim:
                pile_data.pmt[k, index - 1] = value
        elif param_name == "PFI=":
            for k in nim:
                pile_data.pfi[k, index - 1] = value
        elif param_name == "NSG=":
            for k in nim:
                pile_data.nsg[k, index - 1] = int(value)
        elif param_name == "PMB=":
            for k in nim:
                pile_data.pmb[k] = value
        elif param_name == "PEH=":
            for k in nim:
                pile_data.peh[k] = value
        elif param_name == "PKE=":
            for k in nim:
                pile_data.pke[k] = value


def init5(im, is_val, snum, sim_pile_data, element_stiffness, lines, line_index):
    """
    Read and set simulative pile data.
    Equivalent to INIT5 in the original Fortran code.

    Args:
        im: Segment index
        is_val: Starting index for element stiffness
        snum: Number of simulative piles
        sim_pile_data: SimulativePileData object
        element_stiffness: ElementStiffnessData object
        lines: List of file lines
        line_index: Current line index
    """
    if im < 0:
        # Read single line with diagonal stiffness elements
        parts = list(map(float, lines[line_index].strip().split()))
        a = np.array(parts)

        for k in range(snum):
            if sim_pile_data.ksctr[k] == im:
                for ia in range(6):
                    is_val += 1
                    element_stiffness.esp[is_val, :] = 0.0
                    element_stiffness.esp[is_val, ia] = a[ia]

    elif im > 0:
        # Read full 6x6 stiffness matrix
        b = np.zeros((6, 6))
        for i in range(6):
            parts = list(map(float, lines[line_index + i].strip().split()))
            b[i, :] = parts

        for k in range(snum):
            if sim_pile_data.ksctr[k] == im:
                for ia in range(6):
                    is_val += 1
                    element_stiffness.esp[is_val, :] = b[ia, :]


def init6(nact, axy, act):
    """
    Combine external forces.
    Equivalent to INIT6 in the original Fortran code.

    Args:
        nact: Number of actions
        axy: Action coordinates
        act: Action forces

    Returns:
        Combined forces array
    """
    force = np.zeros(6)

    for i in range(nact):
        a = act[i, :]
        tu = tmatx(axy[i, 0], axy[i, 1])
        tn = trnsps(tu)
        b = mulult(tn, a)
        force += b

    return force
