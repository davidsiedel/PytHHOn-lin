from core.domain import Domain
from shapes.triangle import Triangle
from quadratures.dunavant import DunavantRule
from numpy import ndarray as Mat

import numpy as np


class Quadrangle(Domain):
    def __init__(self, vertices: Mat, polynomial_order: int):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Polygon class inherits from the Domain class to specifiy its attributes when the domain is a quadrangle.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the vertices coordinates as vectors.
        - polynomial_order : the polynomial order of integration over the quadrangle.
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - centroid : the vector with values containing the center of mass of the quadrangle.
        - volume : the volume of the quadrangle.
        - diameter : the diameter of the quadrangle.
        - quadrature_points : the matrix containing the quadrature points of the quadrangle.
        - quadrature_weights : the vector containing the quadrature weights of the quadrangle.
        """
        if not vertices.shape == (3, 2):
            raise TypeError("The domain vertices do not match that of a triangle")
        else:
            centroid = Domain.get_domain_barycenter_vector(vertices)
            volume = Quadrangle.get_quadrangle_volume(vertices)
            diameter = Quadrangle.get_quadrangle_diameter(vertices)
            quadrature_points, quadrature_weights = DunavantRule.get_quadrangle_quadrature(
                vertices, volume, polynomial_order
            )
            super().__init__(centroid, volume, diameter, quadrature_points, quadrature_weights)

    @staticmethod
    def get_quadrangle_volume(vertices: Mat) -> float:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the surface of a quadrangle.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the coordinates of the vertices of the quadrangle as vectors.
        ================================================================================================================
        Returns :
        ================================================================================================================
        - quadrangle_volume : the surface of the quadrangle.
        """
        t0 = vertices[0, 1, 2]
        t0 = vertices[0, 3, 2]
        v0 = Triangle.get_triangle_volume(t0)
        v1 = Triangle.get_triangle_volume(t1)
        quadrangle_volume = v0 + v1
        return quadrangle_volume

    @staticmethod
    def get_quadrangle_diameter(vertices: Mat) -> float:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the diamter of a quadrangle.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the coordinates of the vertices of the quadrangle as vectors.
        ================================================================================================================
        Returns :
        ================================================================================================================
        - quadrangle_diameter : the diameter of the quadrangle.
        """
        shape_dimension = 2
        diag0 = vertices[2] - vertices[0]
        diag1 = vertices[3] - vertices[1]
        quadrangle_diameter = max([np.sqrt((diag[1] + diag[0]) ** 2) for diag in [diag0, diag1]])
        return quadrangle_diameter
