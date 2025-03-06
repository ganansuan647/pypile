"""
Pile stiffness calculation module for the BCAD_PILE package.

This module provides functions for calculating deformation factors and stiffness
properties of piles.
"""

import numpy as np
import math
from numba import jit
from ..utils.math_helpers import eaj, parc, param, param1, param2
from ..utils.matrix import mulult, sinver


def calculate_deformation_factors(pnum, zfr, zbl, pile_data):
    """
    Calculate deformation factors of piles.
    Equivalent to BTXY in the original Fortran code.

    Args:
        pnum: Number of piles
        zfr: Array of free lengths
        zbl: Array of buried lengths
        pile_data: PileData object

    Returns:
        Tuple of arrays (btx, bty) containing deformation factors
    """
    btx = np.zeros((pnum, 15))
    bty = np.zeros((pnum, 15))

    # Calculate ground coordinates
    gxy = np.zeros((pnum, 2))
    for k in range(pnum):
        gxy[k, 0] = pile_data.pxy[k, 0] + zfr[k] * pile_data.agl[k, 0]
        gxy[k, 1] = pile_data.pxy[k, 1] + zfr[k] * pile_data.agl[k, 1]

    # Check for closely spaced piles
    for k in range(pnum):
        for k1 in range(k + 1, pnum):
            s = (
                np.sqrt((gxy[k, 0] - gxy[k1, 0]) ** 2 + (gxy[k, 1] - gxy[k1, 1]) ** 2)
                - (pile_data.dob[k, 0] + pile_data.dob[k1, 0]) / 2.0
            )

            if s < 1.0:
                kinf_x = kinf1(1, pnum, pile_data.dob, zbl, gxy)
                kinf_y = kinf1(2, pnum, pile_data.dob, zbl, gxy)
                break
        else:
            continue
        break
    else:
        # No closely spaced piles found
        kinf_x = kinf2(1, pnum, pile_data.dob, zbl, gxy)
        kinf_y = kinf2(2, pnum, pile_data.dob, zbl, gxy)

    # Calculate deformation factors for each pile
    for k in range(pnum):
        if k > 0:
            # Check for piles with the same control value
            for k1 in range(k):
                if pile_data.kctr[k] == pile_data.kctr[k1]:
                    for ia in range(pile_data.nbl[k1]):
                        btx[k, ia] = btx[k1, ia]
                        bty[k, ia] = bty[k1, ia]
                    break
            else:
                # No matching control value found, calculate new factors
                ka = 1.0 if pile_data.ksh[k] == 0 else 0.9
                for ia in range(pile_data.nbl[k]):
                    bx1 = ka * kinf_x * (pile_data.dob[k, ia] + 1.0)
                    by1 = ka * kinf_y * (pile_data.dob[k, ia] + 1.0)

                    a, b = eaj(pile_data.ksh[k], pile_data.pke[k], pile_data.dob[k, ia])

                    btx[k, ia] = (
                        pile_data.pmt[k, ia] * bx1 / (pile_data.peh[k] * b)
                    ) ** 0.2
                    bty[k, ia] = (
                        pile_data.pmt[k, ia] * by1 / (pile_data.peh[k] * b)
                    ) ** 0.2
        else:
            # First pile, always calculate factors
            ka = 1.0 if pile_data.ksh[k] == 0 else 0.9
            for ia in range(pile_data.nbl[k]):
                bx1 = ka * kinf_x * (pile_data.dob[k, ia] + 1.0)
                by1 = ka * kinf_y * (pile_data.dob[k, ia] + 1.0)

                a, b = eaj(pile_data.ksh[k], pile_data.pke[k], pile_data.dob[k, ia])

                btx[k, ia] = (
                    pile_data.pmt[k, ia] * bx1 / (pile_data.peh[k] * b)
                ) ** 0.2
                bty[k, ia] = (
                    pile_data.pmt[k, ia] * by1 / (pile_data.peh[k] * b)
                ) ** 0.2

    return btx, bty


def kinf1(im, pnum, dob, zbl, gxy):
    """
    Calculate influence factor for closely spaced piles.
    Equivalent to KINF1 in the original Fortran code.

    Args:
        im: Direction index (1=x, 2=y)
        pnum: Number of piles
        dob: Array of pile diameters
        zbl: Array of buried lengths
        gxy: Array of ground coordinates

    Returns:
        Influence factor
    """
    # Get unique coordinates in the specified direction
    unique_coords = []
    unique_dobs = []
    unique_zbls = []

    # Add first pile
    unique_coords.append(gxy[0, im - 1])
    unique_dobs.append(dob[0, 0])
    unique_zbls.append(zbl[0])

    # Check other piles
    for k in range(1, pnum):
        if gxy[k, im - 1] not in unique_coords:
            unique_coords.append(gxy[k, im - 1])
            unique_dobs.append(dob[k, 0])
            unique_zbls.append(zbl[k])

    # Calculate influence factor for this unique set
    return kinf3(len(unique_coords), unique_coords, unique_dobs, unique_zbls)


def kinf2(im, pnum, dob, zbl, gxy):
    """
    Calculate minimum influence factor for rows of piles.
    Equivalent to KINF2 in the original Fortran code.

    Args:
        im: Direction index (1=x, 2=y)
        pnum: Number of piles
        dob: Array of pile diameters
        zbl: Array of buried lengths
        gxy: Array of ground coordinates

    Returns:
        Minimum influence factor
    """
    im1 = 2 if im == 1 else 1  # Opposite direction

    # Group piles by coordinates in the perpendicular direction
    groups = {}
    for k in range(pnum):
        coord = gxy[k, im1 - 1]
        if coord in groups:
            groups[coord].append(k)
        else:
            groups[coord] = [k]

    # Calculate influence factor for each group
    kmin = 1.0
    for row_piles in groups.values():
        # Extract coordinates for this row
        aa = [gxy[k, im - 1] for k in row_piles]
        dd = [dob[k, 0] for k in row_piles]
        zz = [zbl[k] for k in row_piles]

        # Calculate influence factor for this row
        kinf = kinf3(len(row_piles), aa, dd, zz)
        kmin = min(kmin, kinf)

    return kmin


def kinf3(n, aa, dd, zz):
    """
    Calculate influence factor for a row of piles.
    Equivalent to KINF3 in the original Fortran code.

    Args:
        n: Number of piles in the row
        aa: Array of pile coordinates
        dd: Array of pile diameters
        zz: Array of pile buried lengths

    Returns:
        Influence factor
    """
    if n == 1:
        return 1.0

    # Calculate effective heights
    ho = np.zeros(n)
    for i in range(n):
        ho[i] = 3.0 * (dd[i] + 1.0)
        if ho[i] > zz[i]:
            ho[i] = zz[i]

    # Find minimum spacing
    lo = 100.0
    hoo = 0.0
    for i in range(n):
        for i1 in range(i + 1, n):
            s = abs(aa[i] - aa[i1]) - (dd[i] + dd[i1]) / 2.0
            if s < lo:
                lo = s
                hoo = max(ho[i], ho[i1])

    # Calculate influence factor
    if lo >= 0.6 * hoo:
        return 1.0
    else:
        c = parc(n)
        return c + (1.0 - c) * lo / (0.6 * hoo)


def calculate_bottom_areas(pnum, zfr, zbl, pile_data):
    """
    Calculate areas at the bottom of piles.
    Equivalent to AREA in the original Fortran code.

    Args:
        pnum: Number of piles
        zfr: Array of free lengths
        zbl: Array of buried lengths
        pile_data: PileData object

    Returns:
        Array of bottom areas
    """
    ao = np.zeros(pnum)

    # Calculate coordinates at pile bottoms
    bxy = np.zeros((pnum, 2))
    for k in range(pnum):
        bxy[k, 0] = pile_data.pxy[k, 0] + (zfr[k] + zbl[k]) * pile_data.agl[k, 0]
        bxy[k, 1] = pile_data.pxy[k, 1] + (zfr[k] + zbl[k]) * pile_data.agl[k, 1]

    # Calculate widths at pile bottoms
    w = np.zeros(pnum)
    for k in range(pnum):
        if pile_data.ksu[k] > 2:
            # For fixed-end piles
            if pile_data.nbl[k] != 0:
                w[k] = pile_data.dob[k, pile_data.nbl[k] - 1]
            else:
                w[k] = pile_data.dof[k, pile_data.nfr[k] - 1]
        else:
            # For free-end piles
            w[k] = 0.0
            ag = np.arctan(np.sqrt(1 - pile_data.agl[k, 2] ** 2) / pile_data.agl[k, 2])

            # Calculate width based on pile angle and soil friction
            for ia in range(pile_data.nbl[k]):
                w[k] += pile_data.hbl[k, ia] * (
                    np.sin(ag)
                    - pile_data.agl[k, 2]
                    * np.tan(ag - pile_data.pfi[k, ia] * np.pi / 720.0)
                )

            w[k] = w[k] * 2 + pile_data.dob[k, 0]

    # Find minimum spacing between pile bottoms
    smin = np.full(pnum, 100.0)
    for k in range(pnum):
        for ia in range(k + 1, pnum):
            s = np.sqrt((bxy[k, 0] - bxy[ia, 0]) ** 2 + (bxy[k, 1] - bxy[ia, 1]) ** 2)
            smin[k] = min(smin[k], s)
            smin[ia] = min(smin[ia], s)

    # Calculate final areas
    for k in range(pnum):
        if smin[k] < w[k]:
            w[k] = smin[k]

        if pile_data.ksh[k] == 0:  # Circular pile
            ao[k] = np.pi * w[k] ** 2 / 4.0
        else:  # Square pile
            ao[k] = w[k] ** 2

    return ao


def calculate_axial_stiffness(pnum, zfr, zbl, ao, pile_data):
    """
    Calculate axial stiffness of piles.
    Equivalent to STIFF_N in the original Fortran code.

    Args:
        pnum: Number of piles
        zfr: Array of free lengths
        zbl: Array of buried lengths
        ao: Array of bottom areas
        pile_data: PileData object

    Returns:
        Array of axial stiffness values
    """
    rzz = np.zeros(pnum)

    # First pile
    rzz[0] = stn(0, zbl[0], ao[0], pile_data)

    # Other piles - check for identical properties
    for k in range(1, pnum):
        # Check if a previous pile has the same properties
        for ia in range(k):
            if pile_data.kctr[k] == pile_data.kctr[ia] and abs(ao[k] - ao[ia]) < 1e-10:
                rzz[k] = rzz[ia]
                break
        else:
            # No matching pile found
            rzz[k] = stn(k, zbl[k], ao[k], pile_data)

    return rzz


def stn(k, zbl, ao, pile_data):
    """
    Calculate axial stiffness of a single pile.
    Equivalent to STN in the original Fortran code.

    Args:
        k: Pile index
        zbl: Buried length
        ao: Bottom area
        pile_data: PileData object

    Returns:
        Axial stiffness value
    """
    # Set coefficient based on support condition
    if pile_data.ksu[k] == 1:
        pkc = 0.5
    elif pile_data.ksu[k] == 2:
        pkc = 0.667
    else:  # pile_data.ksu[k] > 2
        pkc = 1.0

    # Calculate stiffness
    x = 0.0

    # Free segments
    for ia in range(pile_data.nfr[k]):
        a, _ = eaj(pile_data.ksh[k], pile_data.pke[k], pile_data.dof[k, ia])
        x += pile_data.hfr[k, ia] / (pile_data.peh[k] * a)

    # Buried segments
    for ia in range(pile_data.nbl[k]):
        a, _ = eaj(pile_data.ksh[k], pile_data.pke[k], pile_data.dob[k, ia])
        x += pkc * pile_data.hbl[k, ia] / (pile_data.peh[k] * a)

    # Bottom effect
    if pile_data.ksu[k] <= 2:
        x += 1.0 / (pile_data.pmb[k] * zbl * ao)
    else:
        x += 1.0 / (pile_data.pmb[k] * ao)

    # Return stiffness as reciprocal of compliance
    return 1.0 / x


def calculate_lateral_stiffness(pnum, rzz, btx, bty, pile_data, element_stiffness):
    """
    Calculate lateral stiffness of piles.
    Equivalent to PSTIFF in the original Fortran code.

    Args:
        pnum: Number of piles
        rzz: Array of axial stiffness values
        btx: Array of x-direction deformation factors
        bty: Array of y-direction deformation factors
        pile_data: PileData object
        element_stiffness: ElementStiffnessData object

    Returns:
        None (updates element_stiffness.esp directly)
    """
    for k in range(pnum):
        if pile_data.nbl[k] == 0:
            # No buried segments, use identity matrices
            kbx = np.eye(4)
            kby = np.eye(4)
        else:
            # Calculate relation matrices for buried segments
            h = np.zeros(pile_data.nbl[k] + 1)
            ej = np.zeros(pile_data.nbl[k])
            bt1 = np.zeros(pile_data.nbl[k])
            bt2 = np.zeros(pile_data.nbl[k])

            for ia in range(pile_data.nbl[k]):
                bt1[ia] = btx[k, ia]
                bt2[ia] = bty[k, ia]

                _, b = eaj(pile_data.ksh[k], pile_data.pke[k], pile_data.dob[k, ia])
                ej[ia] = pile_data.peh[k] * b

                if ia > 0:
                    h[ia] = h[ia - 1] + pile_data.hbl[k, ia - 1]

            h[pile_data.nbl[k]] = (
                h[pile_data.nbl[k] - 1] + pile_data.hbl[k, pile_data.nbl[k] - 1]
            )

            # Calculate relation matrices
            kbx, kby = rltmtx(pile_data.nbl[k], bt1, bt2, ej, h)

        # Handle free segments
        if pile_data.nfr[k] == 0:
            # No free segments, just use relation matrix with sign change
            kx = kbx.copy()
            ky = kby.copy()

            kx[:, 3] = -kx[:, 3]
            ky[:, 3] = -ky[:, 3]
        else:
            # Calculate relation matrix for free segments
            h = np.zeros(pile_data.nfr[k])
            ej = np.zeros(pile_data.nfr[k])

            for ia in range(pile_data.nfr[k]):
                _, b = eaj(pile_data.ksh[k], pile_data.pke[k], pile_data.dof[k, ia])
                ej[ia] = pile_data.peh[k] * b
                h[ia] = pile_data.hfr[k, ia]

            # Calculate relation matrix for free segments
            kfr = rltfr(pile_data.nfr[k], ej, h)

            # Combine with buried segment matrices
            kx = combx(kbx, kfr)
            ky = combx(kby, kfr)

        # Calculate element stiffness based on boundary conditions
        ke = cndtn(pile_data.ksu[k], kx, ky, rzz[k])

        # Store in global stiffness array
        k1 = k * 6
        for i in range(6):
            for j in range(6):
                element_stiffness.esp[k1 + i, j] = ke[i, j]


def rltmtx(nbl, bt1, bt2, ej, h):
    """
    Calculate relation matrices for non-free pile segments.
    Equivalent to RLTMTX in the original Fortran code.

    Args:
        nbl: Number of buried segments
        bt1: Array of x-direction deformation factors
        bt2: Array of y-direction deformation factors
        ej: Array of flexural rigidities
        h: Array of segment heights

    Returns:
        Tuple of relation matrices (kbx, kby)
    """
    # Calculate x-direction relation matrix
    kbx = saa(bt1[0], ej[0], h[0], h[1])

    for ia in range(1, nbl):
        a1 = kbx.copy()
        a2 = saa(bt1[ia], ej[ia], h[ia], h[ia + 1])
        kbx = np.dot(a2, a1)

    # If deformation factors are the same, use kbx for kby
    if np.allclose(bt1, bt2):
        kby = kbx.copy()
    else:
        # Calculate y-direction relation matrix
        kby = saa(bt2[0], ej[0], h[0], h[1])

        for ia in range(1, nbl):
            a1 = kby.copy()
            a2 = saa(bt2[ia], ej[ia], h[ia], h[ia + 1])
            kby = np.dot(a2, a1)

    return kbx, kby


def saa(bt, ej, h1, h2):
    """
    Calculate relational matrix of one non-free pile segment.
    Equivalent to SAA in the original Fortran code.

    Args:
        bt: Deformation factor
        ej: Flexural rigidity
        h1: Starting height
        h2: Ending height

    Returns:
        4x4 relation matrix
    """
    ai1 = param(bt, ej, h1)
    ai2 = param(bt, ej, h2)
    ai3 = sinver(ai1)

    ai = np.dot(ai2, ai3)

    # Scale certain elements by EJ
    for i in range(2):
        for j in range(2):
            ai[i, j + 2] /= ej
            ai[i + 2, j] *= ej

    # Swap rows and columns 3 and 4
    ai[[2, 3], :] = ai[[3, 2], :]
    ai[:, [2, 3]] = ai[:, [3, 2]]

    return ai


def rltfr(nfr, ej, hfr):
    """
    Calculate relation matrix of free segments of piles.
    Equivalent to RLTFR in the original Fortran code.

    Args:
        nfr: Number of free segments
        ej: Array of flexural rigidities
        hfr: Array of segment heights

    Returns:
        4x4 relation matrix
    """
    # Start with first segment
    kfr = mfree(ej[0], hfr[0])

    # Multiply with subsequent segments
    for ia in range(1, nfr):
        r = mfree(ej[ia], hfr[ia])
        rm = np.dot(kfr, r)
        kfr = rm

    return kfr


def mfree(ej, h):
    """
    Calculate relation matrix of one pile segment.
    Equivalent to MFREE in the original Fortran code.

    Args:
        ej: Flexural rigidity
        h: Segment height

    Returns:
        4x4 relation matrix
    """
    r = np.eye(4)

    r[0, 1] = h
    r[0, 2] = h**3 / (6.0 * ej)
    r[0, 3] = -(h**2) / (2.0 * ej)
    r[1, 2] = h**2 / (2.0 * ej)
    r[1, 3] = -h / ej
    r[3, 2] = -h

    return r


def combx(kbx, kfr):
    """
    Combine relation matrices of free and non-free pile segments.
    Equivalent to COMBX in the original Fortran code.

    Args:
        kbx: Relation matrix for non-free segments
        kfr: Relation matrix for free segments

    Returns:
        Combined relation matrix
    """
    # Change sign of fourth column
    kbx_mod = kbx.copy()
    kbx_mod[:, 3] = -kbx_mod[:, 3]

    # Multiply matrices
    kx = np.dot(kbx_mod, kfr)

    return kx


def cndtn(ksu, kx, ky, rzz):
    """
    Calculate element lateral stiffnesses of a pile.
    Equivalent to CNDTN in the original Fortran code.

    Args:
        ksu: Support condition
        kx: X-direction relation matrix
        ky: Y-direction relation matrix
        rzz: Axial stiffness value

    Returns:
        6x6 element stiffness matrix
    """
    ke = np.zeros((6, 6))

    # Process x-direction
    at_x = dvsn(ksu, kx)
    ke[0, 0] = at_x[0, 0]
    ke[0, 5] = at_x[0, 1]
    ke[5, 0] = at_x[1, 0]
    ke[5, 5] = at_x[1, 1]

    # Process y-direction
    at_y = dvsn(ksu, ky)
    ke[1, 1] = at_y[0, 0]
    ke[1, 4] = -at_y[0, 1]
    ke[4, 1] = -at_y[1, 0]
    ke[4, 4] = at_y[1, 1]

    # Set axial and torsional stiffness
    ke[2, 2] = rzz
    ke[3, 3] = 0.1 * (ke[4, 4] + ke[5, 5])

    return ke


def dvsn(ksu, kxy):
    """
    Apply boundary conditions to pile stiffness.
    Equivalent to DVSN in the original Fortran code.

    Args:
        ksu: Support condition
        kxy: Relation matrix

    Returns:
        2x2 condensed stiffness matrix
    """
    # Extract sub-matrices
    a11 = kxy[:2, :2]
    a12 = kxy[:2, 2:]
    a21 = kxy[2:, :2]
    a22 = kxy[2:, 2:]

    # Apply proper boundary condition
    if ksu == 4:  # Fixed at top
        av = sinver(a12)
        at = np.dot(av, a11)
    else:  # Free at top
        av = sinver(a22)
        at = np.dot(av, a21)

    # Change sign
    at = -at

    return at
