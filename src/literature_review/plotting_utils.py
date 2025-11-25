"""
Plotting utilities for radar_fm visualisation.
Provides standardised plot formatting across the project.
"""

import matplotlib.pyplot as plt
from typing import Optional


def set_favourite_plot_params(ax, labelsize=14, spine_width=2, x_title='x', y_title='y'):
    """
    Apply favourite plot parameters to a given axis.
    
    Args:
        ax: Matplotlib axis object
        labelsize: Font size for labels
        spine_width: Width of axis spines
        x_title: X-axis label
        y_title: Y-axis label
    
    Returns:
        Modified axis object
    """
    if ax is None:
        raise ValueError("Axis object cannot be None")
        
    ax.set_xlabel(x_title, fontsize=labelsize, fontweight='bold')
    ax.set_ylabel(y_title, fontsize=labelsize, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set spine widths
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
        
    ax.tick_params(axis='both', which='major', labelsize=labelsize-2)
    
    return ax


def apply_favourite_figure_params(fig, tight_layout=True):
    """
    Apply figure-level formatting.
    
    Args:
        fig: Matplotlib figure object
        tight_layout: Whether to apply tight_layout (conflicts with subplots_adjust)
    """
    if tight_layout:
        fig.tight_layout()
    else:
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.92, hspace=0.3, wspace=0.3)
    
    return fig


def setup_subplot_with_favourite_params(subplot_pos, x_title='x', y_title='y', 
                                       labelsize=14, spine_width=2):
    """
    Create a subplot and immediately apply favourite parameters.
    
    Args:
        subplot_pos: Subplot position (e.g., (2, 2, 1) or 221)
        x_title: X-axis label
        y_title: Y-axis label
        labelsize: Font size for labels
        spine_width: Width of axis spines
    
    Returns:
        Configured axis object
    """
    if isinstance(subplot_pos, tuple):
        ax = plt.subplot(*subplot_pos)
    else:
        ax = plt.subplot(subplot_pos)
    
    return set_favourite_plot_params(ax, labelsize, spine_width, x_title, y_title)
