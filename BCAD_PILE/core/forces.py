"""
Forces calculation module for the BCAD_PILE package.

This module provides functions for calculating internal forces along pile bodies.
"""

import numpy as np
from ..utils.matrix import mulult
from ..utils.math_helpers import eaj, param


def calculate_pile_forces(pnum, btx, bty, zbl, duk, pile_data, element_stiffness):
    """
    Calculate displacements and internal forces of pile bodies.
    Equivalent to EFORCE in the original Fortran code.

    Args:
        pnum: Number of piles
        btx: X-direction deformation factors
        bty: Y-direction deformation factors
        zbl: Array of buried lengths
        duk: Displacements at pile tops
        pile_data: PileData object
        element_stiffness: ElementStiffnessData object

    Returns:
        List of dictionaries with force data for each pile
    """
    results = []

    for k in range(pnum):
        # Get pile displacements
        ce = duk[k, :]

        # Get pile stiffness
        se = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                se[i, j] = element_stiffness.esp[(k - 1) * 6 + i, j]

        # Calculate forces
        pe = mulult(se, ce)

        # Initialize arrays for section forces
        zh = []
        fx = []  # [ux, sy, nx, my]
        fy = []  # [uy, sx, ny, mx]
        fz = []  # Normal force
        psx = []  # Soil stress in x-direction
        psy = []  # Soil stress in y-direction

        # Initial values at the top
        zh.append(0.0)
        fx.append([ce[0], ce[5], pe[0], pe[5]])
        fy.append([ce[1], ce[4], pe[1], pe[4]])
        fz.append(pe[2])
        psx.append(0.0)
        psy.append(0.0)

        # Calculate forces in free segments
        nsum = 1  # Counter for sections

        for ia in range(pile_data.nfr[k]):
            # Calculate properties
            hl = pile_data.hfr[k, ia] / pile_data.nsf[k, ia]
            a, b = eaj(pile_data.ksh[k], pile_data.pke[k], pile_data.dof[k, ia])
            ej = pile_data.peh[k] * b

            # Initialize relation matrix
            r = np.eye(4)
            r[0, 1] = hl
            r[0, 2] = hl**3 / (6.0 * ej)
            r[0, 3] = -(hl**2) / (2.0 * ej)
            r[1, 2] = hl**2 / (2.0 * ej)
            r[1, 3] = -hl / ej
            r[3, 2] = -hl

            # Process each subdivision
            for _ in range(pile_data.nsf[k, ia]):
                # Get values from previous section
                xa = fx[nsum - 1].copy()
                xc = fy[nsum - 1].copy()

                # Adjust signs
                xc[1] = -xc[1]
                xc[3] = -xc[3]

                # Calculate new values using relation matrix
                xb = mulult(r, xa)
                xd = mulult(r, xc)

                # Increment counter
                nsum += 1

                # Store results with adjusted signs
                zh.append(zh[nsum - 2] + hl)
                fx.append(xb.tolist())
                fy.append([xd[0], -xd[1], xd[2], -xd[3]])
                fz.append(fz[nsum - 2])  # Axial force unchanged in free segment
                psx.append(0.0)
                psy.append(0.0)

        # Remember transition point from free to buried
        ig = nsum
        zg = zh[nsum - 1]

        # Calculate forces in buried segments
        for ia in range(pile_data.nbl[k]):
            # Calculate properties
            hl = pile_data.hbl[k, ia] / pile_data.nsg[k, ia]
            a, b = eaj(pile_data.ksh[k], pile_data.pke[k], pile_data.dob[k, ia])
            ej = pile_data.peh[k] * b

            # Process each subdivision
            for _ in range(pile_data.nsg[k, ia]):
                # Get depth relative to ground level
                h1 = zh[nsum - 1] - zg
                h2 = h1 + hl

                # Get values from previous section
                xa = fx[nsum - 1].copy()
                xc = fy[nsum - 1].copy()

                # Adjust signs
                xa[3] = -xa[3]
                xc[1] = -xc[1]

                # Calculate relation matrices
                r = saa(btx[k, ia], ej, h1, h2)
                xb = mulult(r, xa)

                # Check if different relation matrix needed for y-direction
                if abs(btx[k, ia] - bty[k, ia]) > 1.0e-3:
                    r = saa(bty[k, ia], ej, h1, h2)
                xd = mulult(r, xc)

                # Increment counter
                nsum += 1

                # Store results with adjusted signs
                zh.append(zh[nsum - 2] + hl)
                fx.append([xb[0], xb[1], xb[2], -xb[3]])
                fy.append([xd[0], -xd[1], xd[2], xd[3]])

                # Calculate soil stresses
                psx.append(fx[nsum - 1][0] * h2 * pile_data.pmt[k, ia])
                psy.append(fy[nsum - 1][0] * h2 * pile_data.pmt[k, ia])

                # Calculate axial force distribution
                if pile_data.ksu[k] >= 3:
                    # Fixed-end pile - axial force constant
                    fz.append(fz[ig - 1])
                else:
                    # Free-end pile - axial force decreases with depth
                    fz.append(fz[ig - 1] * (1.0 - h2**2 / zbl[k] ** 2))

        # Add pile results to results list
        pile_result = {
            "pile_number": k + 1,
            "position": [pile_data.pxy[k, 0], pile_data.pxy[k, 1]],
            "top_displacement": ce,
            "top_force": pe,
            "z_coordinates": zh,
            "displacements_x": [f[0] for f in fx],
            "rotations_y": [f[1] for f in fx],
            "shear_x": [f[2] for f in fx],
            "moment_y": [f[3] for f in fx],
            "displacements_y": [f[0] for f in fy],
            "rotations_x": [f[1] for f in fy],
            "shear_y": [f[2] for f in fy],
            "moment_x": [f[3] for f in fy],
            "axial_force": fz,
            "soil_stress_x": psx,
            "soil_stress_y": psy,
            "ground_level_index": ig,
        }

        results.append(pile_result)

    return results


def saa(bt, ej, h1, h2):
    """
    Calculate relational matrix for a non-free pile segment.
    Equivalent to SAA in the original Fortran code.

    Args:
        bt: Deformation factor
        ej: Flexural rigidity
        h1: Starting height
        h2: Ending height

    Returns:
        4x4 relation matrix
    """
    # Calculate parameter matrices
    ai1 = param(bt, ej, h1)
    ai2 = param(bt, ej, h2)

    # Invert first matrix
    ai3 = np.linalg.inv(ai1)

    # Multiply matrices
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
