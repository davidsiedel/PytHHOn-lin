from core.domain import Domain
from quadratures.dunavant import DunavantRule
from numpy import ndarray as Mat
import numpy as np


class Point(Domain):
    def __init__(self, vertices: Mat):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Point class inherits from the Domain class to specifiy its attributes when the domain is a point.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the single valued matrix with value that of the point.
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - centroid : the vector with values containing the center of mass of the point.
        - volume : the volume of the point
        - diameter : the diameter of the point
        - quadrature_points : the matrix containing the quadrature points of the point
        - quadrature_weights : the vector containing the quadrature weights of the point
        """
        if not vertices.shape == (1, 1):
            raise TypeError("The domain vertices do not match that of a point")
        else:
            # centroid = vertices[0]
            centroid = np.array([])
            volume = 1.0
            diameter = 1.0
            quadrature_points, quadrature_weights = DunavantRule.get_point_quadrature()
            super().__init__(centroid, volume, diameter, quadrature_points, quadrature_weights)
