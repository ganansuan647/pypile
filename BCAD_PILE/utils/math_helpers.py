"""
Mathematical helper functions for the BCAD_PILE package.

This module provides implementations of specialized mathematical functions
used in the original Fortran code.
"""

import numpy as np
import math
from numba import jit


def param1(y):
    """
    Calculate parameter values for pile analysis (part 1).
    Equivalent to PARAM1 in the original Fortran code.

    Args:
        y: Input parameter

    Returns:
        Tuple of parameters: (a1, b1, c1, d1, a2, b2, c2, d2)
    """
    a1 = (
        1
        - y**5 / 120.0
        + y**10 / 6.048e5
        - y**15 / 1.9813e10
        + y**20 / 2.3038e15
        - y**25 / 6.9945e20
    )
    b1 = (
        y
        - y**6 / 360.0
        + y**11 / 2851200
        - y**16 / 1.245e11
        + y**21 / 1.7889e16
        - y**26 / 6.4185e21
    )
    c1 = (
        y**2 / 2.0
        - y**7 / 1680
        + y**12 / 1.9958e7
        - y**17 / 1.14e12
        + y**22 / 2.0e17
        - y**27 / 8.43e22
    )
    d1 = (
        y**3 / 6.0
        - y**8 / 10080
        + y**13 / 1.7297e8
        - y**18 / 1.2703e13
        + y**23 / 2.6997e18
        - y**28 / 1.33e24
    )
    a2 = (
        -(y**4) / 24.0
        + y**9 / 6.048e4
        - y**14 / 1.3209e9
        + y**19 / 1.1519e14
        - y**24 / 2.7978e19
    )
    b2 = (
        1
        - y**5 / 60.0
        + y**10 / 2.592e5
        - y**15 / 7.7837e9
        + y**20 / 8.5185e14
        - y**25 / 2.4686e20
    )
    c2 = (
        y
        - y**6 / 240.0
        + y**11 / 1.6632e6
        - y**16 / 6.7059e10
        + y**21 / 9.0973e15
        - y**26 / 3.1222e21
    )
    d2 = (
        y**2 / 2
        - y**7 / 1260
        + y**12 / 1.3305e7
        - y**17 / 7.0572e11
        + y**22 / 1.1738e17
        - y**27 / 4.738e22
    )

    return a1, b1, c1, d1, a2, b2, c2, d2


def param2(y):
    """
    Calculate parameter values for pile analysis (part 2).
    Equivalent to PARAM2 in the original Fortran code.

    Args:
        y: Input parameter

    Returns:
        Tuple of parameters: (a3, b3, c3, d3, a4, b4, c4, d4)
    """
    a3 = (
        -(y**3) / 6
        + y**8 / 6.72e3
        - y**13 / 9.435e7
        + y**18 / 6.0626e12
        - y**23 / 1.1657e18
    )
    b3 = (
        -(y**4) / 12
        + y**9 / 25920
        - y**14 / 5.1892e8
        + y**19 / 4.2593e13
        - y**24 / 9.8746e18
    )
    c3 = (
        1
        - y**5 / 40
        + y**10 / 151200
        - y**15 / 4.1912e9
        + y**20 / 4.332e14
        - y**25 / 1.2009e20
    )
    d3 = (
        y
        - y**6 / 180
        + y**11 / 1108800
        - y**16 / 4.1513e10
        + y**21 / 5.3354e15
        - y**26 / 1.7543e21
    )
    a4 = (
        -(y**2) / 2
        + y**7 / 840
        - y**12 / 7.257e6
        + y**17 / 3.3681e11
        - y**22 / 5.0683e16
    )
    b4 = (
        -(y**3) / 3
        + y**8 / 2880
        - y**13 / 3.7066e7
        + y**18 / 2.2477e12
        - y**23 / 4.1144e17
    )
    c4 = (
        -(y**4) / 8
        + y**9 / 1.512e4
        - y**14 / 2.7941e8
        + y**19 / 2.166e13
        - y**24 / 4.8034e18
    )
    d4 = (
        1
        - y**5 / 30
        + y**10 / 100800
        - y**15 / 2.5946e9
        + y**20 / 2.5406e14
        - y**25 / 6.7491e19
    )

    return a3, b3, c3, d3, a4, b4, c4, d4


def param(bt, ej, x):
    """
    Calculate the coefficient matrix for deformation analysis.
    Equivalent to PARAM in the original Fortran code.

    Args:
        bt: Deformation factor
        ej: Flexural rigidity
        x: Position

    Returns:
        4x4 coefficient matrix (numpy array)
    """
    aa = np.zeros((4, 4))
    y = bt * x
    if y > 6.0:
        y = 6.0

    a1, b1, c1, d1, a2, b2, c2, d2 = param1(y)
    a3, b3, c3, d3, a4, b4, c4, d4 = param2(y)

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

    return aa


def eaj(j, pke, diameter):
    """
    Calculate properties of pile cross section.
    Equivalent to EAJ in the original Fortran code.

    Args:
        j: Pile shape (0 for circular, 1 for square)
        pke: Coefficient for moment of inertia
        diameter: Diameter or width of the pile

    Returns:
        Tuple of (area, moment of inertia)
    """
    if j == 0:  # Circular pile
        area = math.pi * diameter**2 / 4.0
        moment = pke * math.pi * diameter**4 / 64.0
    else:  # Square pile
        area = diameter**2
        moment = pke * diameter**4 / 12.0

    return area, moment


def parc(n):
    """
    Calculate the pile group coefficient.
    Equivalent to PARC in the original Fortran code.

    Args:
        n: Number of piles in a row

    Returns:
        Coefficient value
    """
    if n == 1:
        return 1.0
    elif n == 2:
        return 0.6
    elif n == 3:
        return 0.5
    else:  # n >= 4
        return 0.45


def f_name(filename, extension):
    """
    Create a full filename with extension.
    Equivalent to F_NAME in the original Fortran code.

    Args:
        filename: Base filename
        extension: File extension (including dot)

    Returns:
        Full filename (string)
    """
    # Remove any existing extension or spaces
    base = filename.split(".")[0].strip()
    return f"{base}{extension}"
