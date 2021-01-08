from core.domain import Domain
from quadratures.dunavant import DunavantRule
from numpy import ndarray as Mat

import numpy as np


class Segment(Domain):
    def __init__(self, vertices: Mat, polynomial_order: int):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Segment class inherits from the Domain class to specifiy its attributes when the domain is a segment.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the vertices coordinates as vectors.
        - polynomial_order : the polynomial order of integration over the segment.
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - centroid : the vector with values containing the center of mass of the segment.
        - volume : the volume of the segment.
        - diameter : the diameter of the segment.
        - quadrature_points : the matrix containing the quadrature points of the segment.
        - quadrature_weights : the vector containing the quadrature weights of the segment.
        """
        if not vertices.shape == (2, 1):
            raise TypeError("The domain vertices do not match that of a line")
        else:
            centroid = Domain.get_domain_barycenter_vector(vertices)
            volume = Segment.get_segment_volume(vertices)
            diameter = volume
            quadrature_points, quadrature_weights = DunavantRule.get_segment_quadrature(
                vertices, volume, polynomial_order
            )
            super().__init__(centroid, volume, diameter, quadrature_points, quadrature_weights)

    @staticmethod
    def get_segment_volume(vertices: Mat) -> float:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the length of a segment.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the coordinates of the vertices of the segment as vectors.
        ================================================================================================================
        Returns :
        ================================================================================================================
        - segment_volume : the surface of the segment.
        """
        segment_volume = np.abs(vertices[0, 0] - vertices[1, 0])
        return segment_volume
