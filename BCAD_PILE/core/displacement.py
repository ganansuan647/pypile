"""
Displacement calculation module for the BCAD_PILE package.

This module provides functions for calculating displacements of the pile foundation.
"""

import numpy as np
from ..utils.matrix import mulult, trnsps, tmatx, trnsfr, gaos


def calculate_displacements(jctr, ino, pnum, snum, pile_data, sim_pile_data, 
                          element_stiffness, force, zfr, zbl):
    """
    Calculate displacements of the cap of pile foundation.
    Equivalent to DISP in the original Fortran code.
    
    Args:
        jctr: Control mode (1=full analysis, 2=stiffness only, 3=single pile)
        ino: Pile number for single pile analysis
        pnum: Number of piles
        snum: Number of simulative piles
        pile_data: PileData object
        sim_pile_data: SimulativePileData object
        element_stiffness: ElementStiffnessData object
        force: Force vector
        zfr: Array of free lengths
        zbl: Array of buried lengths
        
    Returns:
        Tuple of (displacement array, stiffness matrix)
    """
    # Initialize stiffness matrix
    so = np.zeros((6, 6))
    
    # Special case for single pile analysis
    if jctr == 3:
        for ia in range(6):
            for ib in range(6):
                so[ia, ib] = element_stiffness.esp[(ino-1)*6 + ia, ib]
        
        return None, so
    
    # Combine all pile stiffnesses
    for k in range(pnum + snum):
        # Get element stiffness matrix
        a = np.zeros((6, 6))
        for ia in range(6):
            for ib in range(6):
                a[ia, ib] = element_stiffness.esp[(k-1)*6 + ia, ib]
        
        # Transform to global coordinates
        if k < pnum:
            # Regular pile
            tk = trnsfr(pile_data.agl[k, 0], pile_data.agl[k, 1], pile_data.agl[k, 2])
            tk1 = trnsps(tk)
            a1 = mulult(tk, a)
            a = mulult(a1, tk1)
            
            x = pile_data.pxy[k, 0]
            y = pile_data.pxy[k, 1]
        else:
            # Simulative pile
            k1 = k - pnum
            x = sim_pile_data.sxy[k1, 0]
            y = sim_pile_data.sxy[k1, 1]
        
        # Apply transformation to cap center
        tu = tmatx(x, y)
        tn = trnsps(tu)
        b = mulult(a, tu)
        a = mulult(tn, b)
        
        # Add to global stiffness
        so += a
    
    # Special case for stiffness-only analysis
    if jctr == 2:
        return None, so
    
    # Calculate displacements using force vector
    displacements = gaos(so, force.copy())
    
    # Calculate displacements at each pile
    duk = np.zeros((pnum, 6))
    for k in range(pnum):
        # Transform global displacements to pile
        tu = tmatx(pile_data.pxy[k, 0], pile_data.pxy[k, 1])
        c1 = mulult(tu, displacements)
        
        # Transform to pile local axis
        tk = trnsfr(pile_data.agl[k, 0], pile_data.agl[k, 1], pile_data.agl[k, 2])
        tk1 = trnsps(tk)
        c = mulult(tk1, c1)
        
        duk[k, :] = c
    
    return duk, so
