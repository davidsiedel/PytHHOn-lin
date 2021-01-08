from core.domain import Domain
from quadratures.dunavant import DunavantRule
from numpy import ndarray as Mat

import numpy as np


class Tetrahedron(Domain):
    def __init__(self, vertices: Mat, polynomial_order: int):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Tetrahedron class inherits from the Domain class to specifiy its attributes when the domain is a
        tetrahedron.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the vertices coordinates as vectors.
        - polynomial_order : the polynomial order of integration over the tetrahedron.
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - centroid : the vector with values containing the center of mass of the tetrahedron.
        - volume : the volume of the tetrahedron.
        - diameter : the diameter of the tetrahedron.
        - quadrature_points : the matrix containing the quadrature points of the tetrahedron.
        - quadrature_weights : the vector containing the quadrature weights of the tetrahedron.
        """
        if not vertices.shape == (4, 3):
            raise TypeError("The domain vertices do not match that of a tetrahedron")
        else:
            barycenter = Domain.get_domain_barycenter_vector(vertices)
            volume = Tetrahedron.get_tetrahedron_volume(vertices)
            diameter = None
            quadrature_nodes, quadrature_weights = DunavantRule.get_tetrahedron_quadrature(
                vertices, volume, polynomial_order
            )
            super().__init__(barycenter, volume, diameter, quadrature_nodes, quadrature_weights)

    @staticmethod
    def get_tetrahedron_volume(vertices: Mat) -> float:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the volume of a tetrahedron.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the coordinates of the vertices of the tetrahedron as vectors.
        ================================================================================================================
        Returns :
        ================================================================================================================
        - tetrahedron_volume : the volume of the tetrahedron.
        """
        shape_dimension = 3
        tetrahedron_origin_vertex_vector = np.tile(vertices[0], (shape_dimension + 1, 1))
        tetrahedron_edges = (vertices - tetrahedron_origin_vertex_vector)[1:]
        tetrahedron_volume = np.abs(1.0 / 6.0 * np.linalg.det(tetrahedron_edges))
        return tetrahedron_volume
