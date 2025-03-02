"""
Interactive visualization module for the BCAD_PILE package.

This module provides an interactive web-based visualization using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_interactive_visualization(vis_data):
    """
    Create an interactive web-based visualization of pile analysis results.
    
    Args:
        vis_data: Dictionary with visualization data
        
    Returns:
        Plotly figure object
    """
    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "scene", "colspan": 2}, None]
        ],
        subplot_titles=("Pile Foundation Layout", "Pile Section Analysis", "3D View (Deformation Exaggerated)"),
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # 1. Add pile plan view (top-left)
    add_plan_view(fig, vis_data, row=1, col=1)
    
    # 2. Add section view for first pile (top-right)
    add_section_view(fig, vis_data, pile_index=0, row=1, col=2)
    
    # 3. Add 3D view (bottom)
    add_3d_view(fig, vis_data, row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title="BCAD_PILE Interactive Analysis Results",
        height=900,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.65,
            xanchor="right",
            x=0.95
        )
    )
    
    # Add interactivity for selecting piles
    add_interactivity(fig, vis_data)
    
    return fig


def add_plan_view(fig, vis_data, row, col):
    """
    Add pile layout plan view to the figure.
    
    Args:
        fig: Plotly figure object
        vis_data: Visualization data dictionary
        row, col: Subplot position
        
    Returns:
        None (modifies figure)
    """
    # Extract pile positions
    pile_positions = vis_data['pile_positions']
    
    # Plot each pile
    x_coords = []
    y_coords = []
    text_labels = []
    
    for pile in pile_positions:
        x, y = pile['x'], pile['y']
        pile_num = pile['pile_number']
        
        x_coords.append(x)
        y_coords.append(y)
        text_labels.append(str(pile_num))
    
    # Add scatter plot for pile positions
    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            marker=dict(
                size=15,
                color='royalblue',
                line=dict(width=1, color='darkblue')
            ),
            text=text_labels,
            textposition="middle center",
            name='Piles',
            customdata=np.arange(len(pile_positions)),  # Store pile indices
            hovertemplate='Pile %{text}<br>Position: (%{x:.2f}, %{y:.2f})<extra></extra>'
        ),
        row=row, col=col
    )
    
    # Calculate axis limits with padding
    if x_coords and y_coords:
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        x_padding = max(2.0, 0.2 * (x_max - x_min))
        y_padding = max(2.0, 0.2 * (y_max - y_min))
        
        fig.update_xaxes(
            range=[x_min - x_padding, x_max + x_padding],
            title="X (m)",
            row=row, col=col
        )
        
        fig.update_yaxes(
            range=[y_min - y_padding, y_max + y_padding],
            title="Y (m)",
            scaleanchor="x",
            scaleratio=1,
            row=row, col=col
        )


def add_section_view(fig, vis_data, pile_index, row, col):
    """
    Add section view for a specific pile.
    
    Args:
        fig: Plotly figure object
        vis_data: Visualization data dictionary
        pile_index: Index of pile to display
        row, col: Subplot position
        
    Returns:
        None (modifies figure)
    """
    # Get pile data
    pile_data = vis_data['pile_data'][pile_index]
    pile_number = pile_data['pile_number']
    
    # Get z coordinates (depth) and ground level index
    z_coords = pile_data['z_coordinates']
    ground_idx = pile_data['ground_level_index']
    
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
    
    # Add traces for each data series
    fig.add_trace(
        go.Scatter(
            x=np.array(ux) * disp_scale,
            y=z_coords,
            mode='lines',
            name=f'UX × {disp_scale}',
            line=dict(color='blue', width=2),
            hovertemplate='Depth: %{y:.2f}m<br>UX: %{x:.4f}m<extra></extra>'
        ),
        row=row, col=col
    )
    
    fig.add_trace(
        go.Scatter(
            x=np.array(uy) * disp_scale,
            y=z_coords,
            mode='lines',
            name=f'UY × {disp_scale}',
            line=dict(color='green', width=2),
            hovertemplate='Depth: %{y:.2f}m<br>UY: %{x:.4f}m<extra></extra>'
        ),
        row=row, col=col
    )
    
    fig.add_trace(
        go.Scatter(
            x=np.array(mx) * moment_scale,
            y=z_coords,
            mode='lines',
            name=f'MX × {moment_scale}',
            line=dict(color='red', width=2),
            hovertemplate='Depth: %{y:.2f}m<br>MX: %{x:.2f}t·m<extra></extra>'
        ),
        row=row, col=col
    )
    
    fig.add_trace(
        go.Scatter(
            x=np.array(my) * moment_scale,
            y=z_coords,
            mode='lines',
            name=f'MY × {moment_scale}',
            line=dict(color='cyan', width=2),
            hovertemplate='Depth: %{y:.2f}m<br>MY: %{x:.2f}t·m<extra></extra>'
        ),
        row=row, col=col
    )
    
    fig.add_trace(
        go.Scatter(
            x=np.array(nz) * axial_scale,
            y=z_coords,
            mode='lines',
            name=f'NZ × {axial_scale}',
            line=dict(color='black', width=2),
            hovertemplate='Depth: %{y:.2f}m<br>NZ: %{x:.2f}t<extra></extra>'
        ),
        row=row, col=col
    )
    
    # Add ground level line if available
    if ground_idx < len(z_coords):
        ground_level = z_coords[ground_idx]
        
        fig.add_trace(
            go.Scatter(
                x=[-0.5, 0.5],  # Will be updated based on actual data range
                y=[ground_level, ground_level],
                mode='lines',
                name='Ground Level',
                line=dict(color='brown', width=2, dash='solid'),
                hoverinfo='none'
            ),
            row=row, col=col
        )
    
    # Update axes
    fig.update_xaxes(
        title="Displacement (m) / Moment (t·m) / Force (t)",
        row=row, col=col
    )
    
    fig.update_yaxes(
        title="Depth (m)",
        autorange="reversed",  # Invert y-axis
        row=row, col=col
    )
    
    # Add title annotation
    fig.add_annotation(
        x=0.75,  # Relative position
        y=0.95,  # Relative position
        xref="paper",
        yref="paper",
        text=f"Pile {pile_number} Section Analysis",
        showarrow=False,
        font=dict(size=14),
        row=row, col=col
    )


def add_3d_view(fig, vis_data, row, col):
    """
    Add 3D view of deformed piles.
    
    Args:
        fig: Plotly figure object
        vis_data: Visualization data dictionary
        row, col: Subplot position
        
    Returns:
        None (modifies figure)
    """
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
    
    # Plot each pile
    for i, pile in enumerate(all_piles):
        # Get pile data
        pos = pile_positions[i]
        pile_x, pile_y = pos['x'], pos['y']
        pile_num = pile['pile_number']
        z_coords = pile['z_coordinates']
        ux = pile['deformation']['ux']
        uy = pile['deformation']['uy']
        
        # Calculate deformed coordinates
        x_coords = pile_x + np.array(ux) * mag_factor
        y_coords = pile_y + np.array(uy) * mag_factor
        
        # Calculate deformation magnitude for coloring
        deform = np.sqrt(np.array(ux)**2 + np.array(uy)**2)
        
        # Plot undeformed pile (grey)
        fig.add_trace(
            go.Scatter3d(
                x=[pile_x, pile_x],
                y=[pile_y, pile_y],
                z=[z_coords[0], z_coords[-1]],
                mode='lines',
                line=dict(color='gray', width=2, dash='dash'),
                name=f'Pile {pile_num} (Undeformed)' if i == 0 else '',
                showlegend=i == 0,
                hoverinfo='none'
            ),
            row=row, col=col
        )
        
        # Plot deformed pile
        fig.add_trace(
            go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='lines',
                line=dict(
                    color=deform,
                    colorscale='Jet',
                    width=5,
                    cmin=0,
                    cmax=max_deform
                ),
                name=f'Pile {pile_num} (Deformed)' if i == 0 else '',
                showlegend=i == 0,
                hovertemplate='Pile %{text}<br>Depth: %{z:.2f}m<br>Deformation: %{customdata:.4f}m<extra></extra>',
                text=[pile_num] * len(z_coords),
                customdata=deform
            ),
            row=row, col=col
        )
        
        # Add pile number at top
        fig.add_trace(
            go.Scatter3d(
                x=[x_coords[0]],
                y=[y_coords[0]],
                z=[z_coords[0]],
                mode='text',
                text=[str(pile_num)],
                textposition="middle center",
                textfont=dict(size=12, color='black'),
                showlegend=False,
                hoverinfo='none'
            ),
            row=row, col=col
        )
    
    # Determine axis limits
    xmin, xmax = min([p['x'] for p in pile_positions]), max([p['x'] for p in pile_positions])
    ymin, ymax = min([p['y'] for p in pile_positions]), max([p['y'] for p in pile_positions])
    
    # Add padding
    x_padding = max(2.0, 0.2 * (xmax - xmin))
    y_padding = max(2.0, 0.2 * (ymax - ymin))
    
    # Set axis properties
    fig.update_scenes(
        xaxis=dict(
            title="X (m)",
            range=[xmin - x_padding, xmax + x_padding]
        ),
        yaxis=dict(
            title="Y (m)",
            range=[ymin - y_padding, ymax + y_padding]
        ),
        zaxis=dict(
            title="Z (m)",
            autorange="reversed"  # Invert z-axis
        ),
        aspectmode='data',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.8),
            up=dict(x=0, y=0, z=1)
        ),
        row=row, col=col
    )
    
    # Add colorbar for deformation
    fig.update_layout(
        scene=dict(
            annotations=[
                dict(
                    showarrow=False,
                    x=0.8,
                    y=0.8,
                    z=0.9,
                    text="Deformation Scale: " + str(mag_factor) + "x",
                    xanchor="left",
                    xshift=10,
                    opacity=0.7,
                    font=dict(size=12, color='black')
                )
            ]
        )
    )


def add_interactivity(fig, vis_data):
    """
    Add interactivity to link pile selection with section view.
    
    Args:
        fig: Plotly figure object
        vis_data: Visualization data dictionary
        
    Returns:
        None (modifies figure)
    """
    # Add hidden traces with stack of data for all piles
    # These will be used to update the section view when a pile is clicked
    for i, pile in enumerate(vis_data['pile_data']):
        pile_number = pile['pile_number']
        z_coords = pile['z_coordinates']
        
        # Store an invisible reference trace with custom data for each pile
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode='markers',
                marker=dict(opacity=0),
                customdata=[{
                    'pile_index': i,
                    'pile_number': pile_number,
                    'z_coords': z_coords,
                    'ux': pile['deformation']['ux'],
                    'uy': pile['deformation']['uy'],
                    'mx': pile['forces']['mx'],
                    'my': pile['forces']['my'],
                    'nz': pile['forces']['nz'],
                    'ground_idx': pile['ground_level_index']
                }],
                name=f'Data_Pile_{pile_number}',
                visible=False,
                showlegend=False,
                hoverinfo='none'
            )
        )
    
    # Configure figure for callbacks
    fig.update_layout(
        clickmode='event+select',
        dragmode='zoom',
        hovermode='closest',
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Update Section View",
                        method="animate",
                        args=[None, {"frame": {"duration": 500, "redraw": True}}]
                    )
                ]
            )
        ]
    )
    
    return fig