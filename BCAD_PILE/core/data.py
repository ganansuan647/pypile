"""
Data structures for the BCAD_PILE package.

This module defines classes that replace the COMMON blocks from the original Fortran code.
"""

import numpy as np


class PileData:
    """
    Class to store non-simulative pile data.
    Replaces the PINF common block from the original Fortran code.
    """

    def __init__(self, max_piles=1000):
        """
        Initialize pile data structures.

        Args:
            max_piles: Maximum number of piles to support
        """
        self.max_piles = max_piles

        # Pile coordinates
        self.pxy = np.zeros((max_piles, 2))

        # Control parameters
        self.kctr = np.zeros(max_piles, dtype=int)
        self.ksh = np.zeros(max_piles, dtype=int)  # Pile shape (0=circular, 1=square)
        self.ksu = np.zeros(max_piles, dtype=int)  # Support conditions

        # Direction cosines
        self.agl = np.zeros((max_piles, 3))

        # Free segment parameters
        self.nfr = np.zeros(max_piles, dtype=int)  # Number of free segments
        self.hfr = np.zeros((max_piles, 15))  # Heights of free segments
        self.dof = np.zeros((max_piles, 15))  # Diameters of free segments
        self.nsf = np.zeros(
            (max_piles, 15), dtype=int
        )  # Subdivisions for free segments

        # Buried segment parameters
        self.nbl = np.zeros(max_piles, dtype=int)  # Number of buried segments
        self.hbl = np.zeros((max_piles, 15))  # Heights of buried segments
        self.dob = np.zeros((max_piles, 15))  # Diameters of buried segments
        self.pmt = np.zeros((max_piles, 15))  # Soil modulus
        self.pfi = np.zeros((max_piles, 15))  # Soil friction angle
        self.nsg = np.zeros(
            (max_piles, 15), dtype=int
        )  # Subdivisions for buried segments

        # Pile material properties
        self.pmb = np.zeros(max_piles)  # Modulus at pile bottom
        self.peh = np.zeros(max_piles)  # Elastic modulus
        self.pke = np.zeros(max_piles)  # Ratio for moment of inertia


class SimulativePileData:
    """
    Class to store simulative pile data.
    Replaces the SIMU common block from the original Fortran code.
    """

    def __init__(self, max_piles=20):
        """
        Initialize simulative pile data structures.

        Args:
            max_piles: Maximum number of simulative piles
        """
        self.max_piles = max_piles
        self.sxy = np.zeros((max_piles, 2))  # Coordinates
        self.ksctr = np.zeros(max_piles, dtype=int)  # Control parameters


class ElementStiffnessData:
    """
    Class to store element stiffness data.
    Replaces the ESTIF common block from the original Fortran code.
    """

    def __init__(self, max_elements=1000000):
        """
        Initialize element stiffness data structures.

        Args:
            max_elements: Maximum number of stiffness elements
        """
        self.max_elements = max_elements
        self.esp = np.zeros((max_elements, 6))  # Element stiffness matrix
