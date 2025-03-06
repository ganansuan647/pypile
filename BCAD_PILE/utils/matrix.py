"""
Matrix operations for the BCAD_PILE package.

This module provides NumPy-based implementations of the matrix operations
used in the original Fortran code.
"""

import numpy as np
from numba import jit


def mulult(a, b):
    """
    Matrix multiplication. Equivalent to MULULT in the original Fortran code.

    Args:
        a: First matrix (numpy array)
        b: Second matrix or vector (numpy array)

    Returns:
        Matrix product a*b (numpy array)
    """
    return np.dot(a, b)


def trnsps(a):
    """
    Matrix transpose. Equivalent to TRNSPS in the original Fortran code.

    Args:
        a: Input matrix (numpy array)

    Returns:
        Transposed matrix (numpy array)
    """
    return np.transpose(a)


def sinver(a):
    """
    Matrix inversion. Equivalent to SINVER in the original Fortran code.

    Args:
        a: Input matrix (numpy array)

    Returns:
        Inverted matrix (numpy array) or None if singular
    """
    n = a.shape[0]

    # Calculate average of diagonal elements for scaling
    sum_diag = np.sum(np.abs(np.diag(a)))
    sum_avg = sum_diag / n

    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        # Matrix is singular or nearly singular
        print("Error: Matrix is singular and cannot be inverted.")
        return None


@jit(nopython=True)
def gaos(a, b):
    """
    Solve linear equation system Ax = b using Gaussian elimination.
    Equivalent to GAOS in the original Fortran code.

    Args:
        a: Coefficient matrix (numpy array)
        b: Right-hand side vector (numpy array)

    Returns:
        Solution vector (numpy array)
    """
    n = len(b)
    a_copy = a.copy()
    b_copy = b.copy()

    # Forward elimination
    for k in range(n):
        # Divide row k by pivot
        t = a_copy[k, k]
        a_copy[k, k:] = a_copy[k, k:] / t
        b_copy[k] = b_copy[k] / t

        # Eliminate below
        for i in range(k + 1, n):
            t = a_copy[i, k]
            for j in range(k + 1, n):
                a_copy[i, j] = a_copy[i, j] - t * a_copy[k, j]
            b_copy[i] = b_copy[i] - t * b_copy[k]

    # Back substitution
    for i1 in range(1, n):
        i = n - i1
        i2 = i + 1
        t = 0.0
        for j in range(i2, n):
            t = t + a_copy[i, j] * b_copy[j]
        b_copy[i] = b_copy[i] - t

    return b_copy


def tmatx(x, y):
    """
    Calculate the transformation matrix for pile coordinates.
    Equivalent to TMATX in the original Fortran code.

    Args:
        x: X-coordinate of pile
        y: Y-coordinate of pile

    Returns:
        6x6 transformation matrix (numpy array)
    """
    tu = np.eye(6)
    tu[0, 5] = -y
    tu[1, 5] = x
    tu[2, 3] = y
    tu[2, 4] = -x
    return tu


def trnsfr(x, y, z):
    """
    Calculate the transformation matrix for pile direction.
    Equivalent to TRNSFR in the original Fortran code.

    Args:
        x, y, z: Direction cosines

    Returns:
        6x6 transformation matrix (numpy array)
    """
    tk = np.zeros((6, 6))
    b = np.sqrt(y**2 + z**2)

    tk[0, 0] = b
    tk[0, 1] = 0.0
    tk[0, 2] = x
    tk[1, 0] = -x * y / b
    tk[1, 1] = z / b
    tk[1, 2] = y
    tk[2, 0] = -x * z / b
    tk[2, 1] = -y / b
    tk[2, 2] = z

    # Copy upper-left 3x3 block to lower-right block
    tk[3:6, 3:6] = tk[0:3, 0:3]

    return tk
