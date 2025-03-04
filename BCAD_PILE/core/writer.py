"""
Output writer module for the BCAD_PILE package.

This module provides functions for writing output files in the format compatible
with the original Fortran code's outputs.
"""

import os
import numpy as np


def write_output_file(filename, jctr, ino, force, so, pile_results):
    """
    Write results to an output file.
    
    Args:
        filename: Output file path
        jctr: Control mode
        ino: Pile number for single pile analysis
        force: Force vector
        so: Stiffness matrix
        pile_results: List of pile force results
        
    Returns:
        None
    """
    with open(filename, 'w') as f:
        # Write header
        write_header(f)
        
        # Special case for stiffness-only analysis
        if jctr == 2:
            f.write("\n\n" + " " * 7 + "*** Stiffness of the entire pile foundation ***\n\n")
            write_matrix(f, so)
            return
        
        # Special case for single pile analysis
        if jctr == 3:
            f.write("\n\n" + " " * 7 + f"*** Stiffness of the No.{ino} pile ***\n\n")
            write_matrix(f, so)
            return
        
        # Full analysis
        f.write("\n" + "=" * 80 + "\n")
        f.write(" " * 15 + "DISPLACEMENTS AT THE CAP CENTER OF PILE FOUNDATION\n")
        f.write("=" * 80 + "\n")
        
        # Write displacement results
        f.write(f"\n{' ' * 16}Movement in the direction of X axis : UX= {force[0]:.4E} (m)\n")
        f.write(f"{' ' * 16}Movement in the direction of Y axis : UY= {force[1]:.4E} (m)\n")
        f.write(f"{' ' * 16}Movement in the direction of Z axis : UZ= {force[2]:.4E} (m)\n")
        f.write(f"{' ' * 16}Rotational angle  around X axis :     SX= {force[3]:.4E} (rad)\n")
        f.write(f"{' ' * 16}Rotational angle around Y axis :      SY= {force[4]:.4E} (rad)\n")
        f.write(f"{' ' * 16}Rotational angle around Z axis :      SZ= {force[5]:.4E} (rad)\n\n")
        
        # Write results for each pile
        for pile in pile_results:
            # Pile header
            f.write("=" * 80 + "\n")
            f.write(" " * 34 + f"NO. {pile['pile_number']} # PILE\n")
            f.write("=" * 80 + "\n")
            
            # Pile position
            f.write(f"\n{' ' * 12}Coordinator of the pile: (x,y) = ({pile['position'][0]:.4E} ,{pile['position'][1]:.4E} )\n\n")
            
            # Top displacements and forces
            f.write(f"{' ' * 12}Displacements and internal forces at the top of pile:\n\n")
            f.write(f"{' ' * 15}UX= {pile['top_displacement'][0]:.4E} (m){' ' * 9}NX= {pile['top_force'][0]:.4E} (t)\n")
            f.write(f"{' ' * 15}UY= {pile['top_displacement'][1]:.4E} (m){' ' * 9}NY= {pile['top_force'][1]:.4E} (t)\n")
            f.write(f"{' ' * 15}UZ= {pile['top_displacement'][2]:.4E} (m){' ' * 9}NZ= {pile['top_force'][2]:.4E} (t)\n")
            f.write(f"{' ' * 15}SX= {pile['top_displacement'][3]:.4E} (rad){' ' * 7}MX= {pile['top_force'][3]:.4E} (t*m)\n")
            f.write(f"{' ' * 15}SY= {pile['top_displacement'][4]:.4E} (rad){' ' * 7}MY= {pile['top_force'][4]:.4E} (t*m)\n")
            f.write(f"{' ' * 15}SZ= {pile['top_displacement'][5]:.4E} (rad){' ' * 7}MZ= {pile['top_force'][5]:.4E} (t*m)\n\n")
            
            # Displacement table header
            f.write("%" * 80 + "\n")
            f.write(" " * 32 + "Displacements of the pile body and\n")
            f.write(" " * 27 + "   Compression stresses of soil (PSX,PSY)\n")
            f.write("%" * 80 + "\n")
            
            # Displacement table
            f.write("\n" + " " * 15 + "Z" + " " * 12 + "UX" + " " * 12 + "UY" + " " * 12 + "SX" + " " * 12 + "SY" + " " * 12 + "PSX" + " " * 12 + "PSY\n")
            f.write(" " * 14 + "(m)" + " " * 11 + "(m)" + " " * 11 + "(m)" + " " * 10 + "(rad)" + " " * 9 + "(rad)" + " " * 9 + "(t/m2)" + " " * 9 + "(t/m2)\n\n")
            
            # Displacement table data
            ig = pile['ground_level_index']
            num_points = len(pile['z_coordinates'])
            
            # Free segments (above ground)
            for i in range(ig):
                f.write(f"{' ' * 7}{pile['z_coordinates'][i]:14.4E}{pile['displacements_x'][i]:14.4E}{pile['displacements_y'][i]:14.4E}{pile['rotations_x'][i]:14.4E}{pile['rotations_y'][i]:14.4E}\n")
            
            # Buried segments (below ground)
            for i in range(ig, num_points):
                f.write(f"{' ' * 7}{pile['z_coordinates'][i]:14.4E}{pile['displacements_x'][i]:14.4E}{pile['displacements_y'][i]:14.4E}{pile['rotations_x'][i]:14.4E}{pile['rotations_y'][i]:14.4E}{pile['soil_stress_x'][i]:14.4E}{pile['soil_stress_y'][i]:14.4E}\n")
            
            # Forces table header
            f.write("\n\n")
            f.write("%" * 80 + "\n")
            f.write(" " * 32 + "Internal forces of the pile body\n")
            f.write("%" * 80 + "\n")
            
            # Forces table
            f.write("\n" + " " * 18 + "Z" + " " * 14 + "NX" + " " * 14 + "NY" + " " * 14 + "NZ" + " " * 14 + "MX" + " " * 14 + "MY\n")
            f.write(" " * 17 + "(m)" + " " * 12 + "(t)" + " " * 13 + "(t)" + " " * 13 + "(t)" + " " * 12 + "(t*m)" + " " * 11 + "(t*m)\n\n")
            
            # Forces table data
            for i in range(num_points):
                f.write(f"{' ' * 7}{pile['z_coordinates'][i]:16.4E}{pile['shear_x'][i]:16.4E}{pile['shear_y'][i]:16.4E}{pile['axial_force'][i]:16.4E}{pile['moment_x'][i]:16.4E}{pile['moment_y'][i]:16.4E}\n")
            
            f.write("\n\n")


def write_position_file(filename, pile_results):
    """
    Write pile position and results to a position file for visualization.
    
    Args:
        filename: Position file path
        pile_results: List of pile force results
        
    Returns:
        None
    """
    with open(filename, 'w') as f:
        # Write number of piles
        num_piles = len(pile_results)
        f.write(f"{num_piles:5d}\n")
        
        # Write pile positions
        for pile in pile_results:
            f.write(f"{pile['position'][0]:14.4E}{pile['position'][1]:14.4E}\n")
        
        # Write detailed data for each pile
        for pile in pile_results:
            num_points = len(pile['z_coordinates'])
            
            # Write pile number and number of points
            f.write(f"{pile['pile_number']:5d}{num_points:5d}\n")
            
            # Write pile position
            f.write(f"{pile['position'][0]:14.4E}{pile['position'][1]:14.4E}\n")
            
            # Write z coordinates
            for i in range(0, num_points, 6):
                end = min(i+6, num_points)
                line = "".join([f"{pile['z_coordinates'][j]:14.4E}" for j in range(i, end)])
                f.write(line + "\n")
            
            # Write x displacements and forces
            for j in range(4):
                data = [pile['displacements_x'], pile['rotations_y'], pile['shear_x'], pile['moment_y']][j]
                for i in range(0, num_points, 6):
                    end = min(i+6, num_points)
                    line = "".join([f"{data[k]:14.4E}" for k in range(i, end)])
                    f.write(line + "\n")
            
            # Write y displacements and forces
            for j in range(4):
                data = [pile['displacements_y'], pile['rotations_x'], pile['shear_y'], pile['moment_x']][j]
                for i in range(0, num_points, 6):
                    end = min(i+6, num_points)
                    line = "".join([f"{data[k]:14.4E}" for k in range(i, end)])
                    f.write(line + "\n")
            
            # Write axial forces
            for i in range(0, num_points, 6):
                end = min(i+6, num_points)
                line = "".join([f"{pile['axial_force'][j]:14.4E}" for j in range(i, end)])
                f.write(line + "\n")
            
            # Write soil stresses
            for j in range(2):
                data = [pile['soil_stress_x'], pile['soil_stress_y']][j]
                for i in range(0, num_points, 6):
                    end = min(i+6, num_points)
                    line = "".join([f"{data[k]:14.4E}" for k in range(i, end)])
                    f.write(line + "\n")


def write_header(f):
    """
    Write the program header to the output file.
    
    Args:
        f: File object
        
    Returns:
        None
    """
    f.write("\n" * 6)
    f.write("+" * 80 + "\n")
    f.write("+" + " " * 78 + "+\n")
    f.write("+    BBBBBB       CCCC        A       DDDDD         PPPPPP     III     L         EEEEEEE    +\n")
    f.write("+    B     B     C    C      A A      D    D        P     P     I      L         E          +\n")
    f.write("+    B     B    C           A   A     D     D       P     P     I      L         E          +\n")
    f.write("+    BBBBBB     C          A     A    D     D       PPPPPP      I      L         EEEEEEE    +\n")
    f.write("+    B     B    C          AAAAAAA    D     D       P           I      L         E          +\n")
    f.write("+    B     B     C    C    A     A    D    D        P           I      L     L   E          +\n")
    f.write("+    BBBBBB       CCCC     A     A    DDDDD   ===== P          III      LLLLL    EEEEEEE    +\n")
    f.write("+" + " " * 78 + "+\n")
    f.write("+                            Copyright 1990, Python Version 2023                           +\n")
    f.write("+" + " " * 78 + "+\n")
    f.write("+" * 80 + "\n")
    f.write("\n" + " " * 15 + "Welcome to use the BCAD_PILE program !!\n\n")
    f.write(" " * 15 + "This program is aimed to execute spatial statical analysis of pile\n")
    f.write(" " * 15 + "foundations of bridge substructures. If you have any questions about\n")
    f.write(" " * 15 + "this program, please do not hesitate to write to :\n\n")
    f.write(" " * 60 + "CAD Reseach Group\n")
    f.write(" " * 60 + "Dept.of Bridge Engr.\n")
    f.write(" " * 60 + "Tongji University\n")
    f.write(" " * 60 + "1239 Sipin Road \n")
    f.write(" " * 60 + "Shanghai 200092\n")
    f.write(" " * 60 + "P.R.of China\n")


def write_matrix(f, matrix):
    """
    Write a matrix to the output file.
    
    Args:
        f: File object
        matrix: 2D numpy array
        
    Returns:
        None
    """
    for i in range(matrix.shape[0]):
        line = "".join([f"{matrix[i,j]:12.4E}" for j in range(matrix.shape[1])])
        f.write(line + "\n")
