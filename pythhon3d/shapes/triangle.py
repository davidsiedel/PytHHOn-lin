from core.domain import Domain
from quadratures.dunavant import DunavantRule
from numpy import ndarray as Mat

import numpy as np


class Triangle(Domain):
    def __init__(self, vertices: Mat, polynomial_order: int):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Triangle class inherits from the Domain class to specifiy its attributes when the domain is a triangle.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the vertices coordinates as vectors.
        - polynomial_order : the polynomial order of integration over the triangle.
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - centroid : the vector with values containing the center of mass of the triangle.
        - volume : the volume of the triangle.
        - diameter : the diameter of the triangle.
        - quadrature_points : the matrix containing the quadrature points of the triangle.
        - quadrature_weights : the vector containing the quadrature weights of the triangle.
        """
        if not vertices.shape == (3, 2):
            raise TypeError("The domain vertices do not match that of a triangle")
        else:
            centroid = Domain.get_domain_barycenter_vector(vertices)
            volume = Triangle.get_triangle_volume(vertices)
            diameter = Triangle.get_triangle_diameter(vertices)
            quadrature_nodes, quadrature_weights = DunavantRule.get_triangle_quadrature(
                vertices, volume, polynomial_order
            )
            super().__init__(centroid, volume, diameter, quadrature_nodes, quadrature_weights)

    @staticmethod
    def get_triangle_volume(vertices: Mat) -> float:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the surface of a triangle.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the coordinates of the vertices of the triangle as vectors.
        ================================================================================================================
        Returns :
        ================================================================================================================
        - triangle_volume : the surface of the triangle.
        """
        shape_dimension = 2
        triangle_origin_vertex_vector = np.tile(vertices[0], (shape_dimension + 1, 1))
        triangle_edges = (vertices - triangle_origin_vertex_vector)[1:]
        triangle_volume = np.abs(1.0 / 2.0 * np.linalg.det(triangle_edges))
        return triangle_volume

    @staticmethod
    def get_triangle_diameter(vertices: Mat) -> float:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the diameter of a triangle.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the coordinates of the vertices of the triangle as vectors.
        ================================================================================================================
        Returns :
        ================================================================================================================
        - triangle_diameter : the diameter of the triangle.
        """
        shape_dimension = 2
        triangle_origin_vertex_vector = np.tile(vertices[0], (shape_dimension + 1, 1))
        triangle_edges = (vertices - triangle_origin_vertex_vector)[1:]
        triangle_diameter = max([np.sqrt((e[1] + e[0]) ** 2) for e in triangle_edges])
        e_0 = vertices[1] - vertices[0]
        e_1 = vertices[2] - vertices[1]
        e_2 = vertices[2] - vertices[0]
        edges = [e_0, e_1, e_2]
        triangle_diameter = max([np.sqrt(e[0] ** 2 + e[1] ** 2) for e in edges])
        return triangle_diameter
