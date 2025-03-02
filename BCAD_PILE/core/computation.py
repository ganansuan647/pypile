"""
Main computation module for the BCAD_PILE package.

This module provides the high-level functions for running the pile analysis.
"""

import os
import numpy as np
from .data import PileData, SimulativePileData, ElementStiffnessData
from .reader import read_input_file
from .stiffness import (
    calculate_deformation_factors, 
    calculate_bottom_areas, 
    calculate_axial_stiffness,
    calculate_lateral_stiffness
)
from .displacement import calculate_displacements
from .forces import calculate_pile_forces
from .writer import write_output_file, write_position_file
from ..utils.math_helpers import f_name


def analyze_pile_foundation(input_file):
    """
    Perform spatial statical analysis of pile foundations.
    
    Args:
        input_file: Path to input file
        
    Returns:
        Dictionary with analysis results
    """
    # Initialize data structures
    pile_data = PileData()
    sim_pile_data = SimulativePileData()
    element_stiffness = ElementStiffnessData()
    
    # Parse the base filename
    base_file = os.path.basename(input_file)
    base_name = os.path.splitext(base_file)[0]
    
    # Generate output filenames
    output_file = f_name(base_name, '.out')
    position_file = f_name(base_name, '.pos')
    
    # Read input data
    jctr, ino, pnum, snum, force, zfr, zbl = read_input_file(
        input_file, pile_data, sim_pile_data, element_stiffness
    )
    
    # Step 1: Calculate deformation factors
    print("*** Calculating deformation factors of piles ***")
    btx, bty = calculate_deformation_factors(pnum, zfr, zbl, pile_data)
    
    # Step 2: Calculate areas at the bottom of piles
    print("*** Calculating areas at the bottom of piles ***")
    ao = calculate_bottom_areas(pnum, zfr, zbl, pile_data)
    
    # Step 3: Calculate axial stiffness
    print("*** Calculating axial stiffness of piles ***")
    rzz = calculate_axial_stiffness(pnum, zfr, zbl, ao, pile_data)
    
    # Step 4: Calculate lateral stiffness
    print("*** Calculating lateral stiffness of piles ***")
    calculate_lateral_stiffness(pnum, rzz, btx, bty, pile_data, element_stiffness)
    
    # Step 5: Calculate displacements
    print("*** Executing entire pile foundation analysis ***")
    duk, so = calculate_displacements(
        jctr, ino, pnum, snum, pile_data, sim_pile_data, element_stiffness, force, zfr, zbl
    )
    
    # Stop here for stiffness-only or single-pile analysis
    if jctr in [2, 3]:
        write_output_file(output_file, jctr, ino, force, so, None)
        
        results = {
            'jctr': jctr,
            'stiffness_matrix': so,
            'pile_results': None
        }
        
        if jctr == 2:
            print(f"Stiffness matrix written to {output_file}")
        else:
            print(f"Stiffness of pile {ino} written to {output_file}")
        
        return results
    
    # Step 6: Calculate forces along pile bodies
    pile_results = calculate_pile_forces(pnum, btx, bty, zbl, duk, pile_data, element_stiffness)
    
    # Step 7: Write output files
    write_output_file(output_file, jctr, ino, force, so, pile_results)
    write_position_file(position_file, pile_results)
    
    print(f"Results written to {output_file} and {position_file}")
    
    # Prepare result dictionary
    results = {
        'jctr': jctr,
        'force': force,
        'stiffness_matrix': so,
        'pile_results': pile_results,
        'output_file': output_file,
        'position_file': position_file
    }
    
    return results


def extract_visualization_data(results):
    """
    Extract data from analysis results for visualization.
    
    Args:
        results: Analysis results dictionary
        
    Returns:
        Dictionary with visualization data
    """
    if results['jctr'] in [2, 3] or results['pile_results'] is None:
        return None
    
    # Extract global displacement data
    global_disp = {
        'ux': results['force'][0],
        'uy': results['force'][1],
        'uz': results['force'][2],
        'rx': results['force'][3],
        'ry': results['force'][4],
        'rz': results['force'][5]
    }
    
    # Extract pile positions
    pile_positions = []
    for pile in results['pile_results']:
        pile_positions.append({
            'pile_number': pile['pile_number'],
            'x': pile['position'][0],
            'y': pile['position'][1]
        })
    
    # Extract pile deformation data
    pile_data = []
    for pile in results['pile_results']:
        pile_data.append({
            'pile_number': pile['pile_number'],
            'z_coords': pile['z_coordinates'],
            'ground_level_index': pile['ground_level_index'],
            'deformation': {
                'ux': pile['displacements_x'],
                'uy': pile['displacements_y'],
                'rx': pile['rotations_x'],
                'ry': pile['rotations_y']
            },
            'forces': {
                'nx': pile['shear_x'],
                'ny': pile['shear_y'],
                'nz': pile['axial_force'],
                'mx': pile['moment_x'],
                'my': pile['moment_y']
            },
            'soil_stress': {
                'psx': pile['soil_stress_x'],
                'psy': pile['soil_stress_y']
            }
        })
    
    vis_data = {
        'global_displacement': global_disp,
        'pile_positions': pile_positions,
        'pile_data': pile_data,
        'stiffness_matrix': results['stiffness_matrix'].tolist()
    }
    
    return vis_data