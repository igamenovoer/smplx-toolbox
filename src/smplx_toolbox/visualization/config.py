"""Configuration for SMPL visualization.

This module provides simple configuration classes for default colors and styles
used in SMPL visualization.

Classes
-------
VisualizationConfig : Simple configuration for default colors and styles
"""

from __future__ import annotations


class VisualizationConfig:
    """Simple configuration for default colors and styles.

    Provides default colors for mesh, joint, and bone visualization.
    All colors are RGB tuples with values in range [0, 1].

    Attributes
    ----------
    mesh_color : tuple[float, float, float]
        Default color for mesh visualization
    joint_color : tuple[float, float, float]
        Default color for joint visualization
    bone_color : tuple[float, float, float]
        Default color for bone/skeleton visualization

    Examples
    --------
    >>> config = VisualizationConfig()
    >>> print(config.mesh_color)
    (0.8, 0.8, 0.9)

    >>> # Customize colors
    >>> config = VisualizationConfig()
    >>> config.m_mesh_color = (1.0, 0.5, 0.5)
    """

    def __init__(self) -> None:
        """Initialize with sensible defaults.

        Sets default colors for mesh (light gray-blue), joints (red),
        and bones (blue).
        """
        # Default colors as RGB tuples
        self.m_mesh_color: tuple[float, float, float] = (0.8, 0.8, 0.9)
        self.m_joint_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
        self.m_bone_color: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def mesh_color(self) -> tuple[float, float, float]:
        """Get default mesh color.

        Returns
        -------
        tuple[float, float, float]
            RGB color values in range [0, 1]
        """
        return self.m_mesh_color

    @property
    def joint_color(self) -> tuple[float, float, float]:
        """Get default joint color.

        Returns
        -------
        tuple[float, float, float]
            RGB color values in range [0, 1]
        """
        return self.m_joint_color

    @property
    def bone_color(self) -> tuple[float, float, float]:
        """Get default bone color.

        Returns
        -------
        tuple[float, float, float]
            RGB color values in range [0, 1]
        """
        return self.m_bone_color
