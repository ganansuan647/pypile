"""
Visualization module for the BCAD_PILE package.

This module provides functions for creating plots of pile analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


def plot_results(vis_data):
    """
    Create visualization plots for pile analysis results.
    
    Args:
        vis_data: Dictionary with visualization data
        
    Returns:
        None (displays plots)
    """
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot layout:
    # 1. Pile plan view
    # 2. Force/displacement along a selected pile
    # 3. 3D view of deformed piles
    
    # 1. Pile plan view (top-left)
    ax1 = fig.add_subplot(2, 2, 1)
    plot_pile_plan(ax1, vis_data)
    
    # 2. Force/displacement along selected pile (top-right)
    ax2 = fig.add_subplot(2, 2, 2)
    # Initially plot first pile
    selected_pile = 0
    plot_pile_section(ax2, vis_data, selected_pile)
    
    # 3. 3D view of deformed piles (bottom)
    ax3 = fig.add_subplot(2, 2, (3, 4), projection='3d')
    plot_3d_piles(ax3, vis_data)
    
    # Title for the entire figure
    plt.suptitle('BCAD_PILE Analysis Results', fontsize=16)
    
    # Add interactive elements for selecting piles
    def on_click(event):
        if event.inaxes == ax1:
            # Find the closest pile to the click
            pile_positions = vis_data['pile_positions']
            min_dist = float('inf')
            closest_pile = 0
            
            for i, pos in enumerate(pile_positions):
                dist = np.sqrt((event.xdata - pos['x'])**2 + (event.ydata - pos['y'])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_pile = i
            
            # Update the section plot
            ax2.clear()
            plot_pile_section(ax2, vis_data, closest_pile)
            fig.canvas.draw_idle()
    
    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_pile_plan(ax, vis_data):
    """
    Plot pile layout in plan view.
    
    Args:
        ax: Matplotlib axes object
        vis_data: Visualization data dictionary
        
    Returns:
        None (modifies axes object)
    """
    # Extract pile positions
    pile_positions = vis_data['pile_positions']
    
    # Setup axes
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Pile Foundation Layout (Plan View)')
    
    # Plot each pile
    for pile in pile_positions:
        x, y = pile['x'], pile['y']
        pile_num = pile['pile_number']
        
        # Draw pile
        circle = Circle((x, y), 0.5, fill=True, color='lightblue', alpha=0.7)
        ax.add_patch(circle)
        
        # Add pile number
        ax.text(x, y, str(pile_num), ha='center', va='center', fontweight='bold')
    
    # Determine axis limits with some padding
    x_coords = [pile['x'] for pile in pile_positions]
    y_coords = [pile['y'] for pile in pile_positions]
    
    if x_coords and y_coords:
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding (20% of range or at least 2m)
        x_padding = max(2.0, 0.2 * (x_max - x_min))
        y_padding = max(2.0, 0.2 * (y_max - y_min))
        
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)


def plot_pile_section(ax, vis_data, pile_index):
    """
    Plot force and displacement along a pile.
    
    Args:
        ax: Matplotlib axes object
        vis_data: Visualization data dictionary
        pile_index: Index of pile to plot
        
    Returns:
        None (modifies axes object)
    """
    # Get pile data
    pile_data = vis_data['pile_data'][pile_index]
    pile_number = pile_data['pile_number']
    
    # Setup axes
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Displacement (m) / Moment (t·m) / Force (t)')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'Pile {pile_number} - Section Analysis')
    
    # Get z coordinates (depth) and ground level index
    z_coords = pile_data['z_coordinates']
    ground_idx = pile_data['ground_level_index']
    
    # Plot ground line
    if ground_idx < len(z_coords):
        ground_level = z_coords[ground_idx]
        ax.axhline(y=ground_level, color='brown', linestyle='-', linewidth=2, alpha=0.7)
        ax.text(-0.1, ground_level, 'Ground', va='bottom', ha='right')
    
    # Invert y-axis for depth
    ax.invert_yaxis()
    
    # Get displacement and force data
    ux = pile_data['deformation']['ux']
    uy = pile_data['deformation']['uy']
    mx = pile_data['forces']['mx']
    my = pile_data['forces']['my']
    nz = pile_data['forces']['nz']
    
    # Scale factors to make different quantities visible on same plot
    disp_scale = 1.0
    moment_scale = 0.1
    axial_scale = 0.05
    
    # Plot lateral displacements
    ax.plot(np.array(ux) * disp_scale, z_coords, 'b-', label='UX × ' + str(disp_scale))
    ax.plot(np.array(uy) * disp_scale, z_coords, 'g-', label='UY × ' + str(disp_scale))
    
    # Plot bending moments
    ax.plot(np.array(mx) * moment_scale, z_coords, 'r-', label='MX × ' + str(moment_scale))
    ax.plot(np.array(my) * moment_scale, z_coords, 'c-', label='MY × ' + str(moment_scale))
    
    # Plot axial force
    ax.plot(np.array(nz) * axial_scale, z_coords, 'k-', label='NZ × ' + str(axial_scale))
    
    ax.legend(loc='best')


def plot_3d_piles(ax, vis_data):
    """
    Create a 3D visualization of deformed piles.
    
    Args:
        ax: Matplotlib 3D axes object
        vis_data: Visualization data dictionary
        
    Returns:
        None (modifies axes object)
    """
    # Setup axes
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D View of Pile Foundation (Deformation Exaggerated)')
    
    # Collect all pile data
    all_piles = vis_data['pile_data']
    pile_positions = vis_data['pile_positions']
    
    # Magnification factor for deformations
    mag_factor = 20.0
    
    # Get limits for colormap
    max_deform = 0
    for pile in all_piles:
        max_pile_deform = max(
            max(np.abs(pile['deformation']['ux'])), 
            max(np.abs(pile['deformation']['uy']))
        )
        max_deform = max(max_deform, max_pile_deform)
    
    # Create colormap for deformation
    cmap = cm.get_cmap('jet')
    
    # Plot each pile
    for i, pile in enumerate(all_piles):
        # Get pile data
        pos = pile_positions[i]
        pile_x, pile_y = pos['x'], pos['y']
        z_coords = pile['z_coordinates']
        ux = pile['deformation']['ux']
        uy = pile['deformation']['uy']
        
        # Calculate deformed coordinates
        x_coords = pile_x + np.array(ux) * mag_factor
        y_coords = pile_y + np.array(uy) * mag_factor
        
        # Calculate deformation magnitude for coloring
        deform = np.sqrt(np.array(ux)**2 + np.array(uy)**2)
        colors = [cmap(min(d / max_deform, 1.0)) for d in deform]
        
        # Plot undeformed pile (grey)
        ax.plot([pile_x, pile_x], [pile_y, pile_y], [z_coords[0], z_coords[-1]], 
                'gray', alpha=0.3, linestyle='--')
        
        # Plot deformed pile with color gradient
        for j in range(len(z_coords) - 1):
            ax.plot([x_coords[j], x_coords[j+1]], 
                    [y_coords[j], y_coords[j+1]], 
                    [z_coords[j], z_coords[j+1]], 
                    color=colors[j], linewidth=2)
        
        # Add pile number at top
        ax.text(x_coords[0], y_coords[0], z_coords[0], str(pile['pile_number']), 
                ha='center', va='center', fontweight='bold')
    
    # Determine axis limits
    xmin, xmax = min([p['x'] for p in pile_positions]), max([p['x'] for p in pile_positions])
    ymin, ymax = min([p['y'] for p in pile_positions]), max([p['y'] for p in pile_positions])
    
    # Add padding
    x_padding = max(2.0, 0.2 * (xmax - xmin))
    y_padding = max(2.0, 0.2 * (ymax - ymin))
    
    ax.set_xlim(xmin - x_padding, xmax + x_padding)
    ax.set_ylim(ymin - y_padding, ymax + y_padding)
    
    # Set a reasonable zlim (invert z-axis)
    max_z = max([max(pile['z_coordinates']) for pile in all_piles])
    min_z = min([min(pile['z_coordinates']) for pile in all_piles])
    ax.set_zlim(max_z, min_z)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_deform))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Deformation Magnitude (m)')
    
    # Set view angle
    ax.view_init(elev=30, azim=45)