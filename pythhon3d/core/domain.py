from typing import List
from numpy import ndarray as Mat
import numpy as np


class Domain:
    def __init__(
        self, centroid: Mat, volume: float, diameter: float, quadrature_points: List[Mat], quadrature_weights: List[Mat]
    ):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Domain class builds the framework to specify the attributes of a domain in the euclidian space.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - centroid : the vector with values containing the center of mass of the domain.
        - volume : the volume of the domain
        - diameter : the diameter of the domain
        - quadrature_points : the matrix containing the quadrature points of the domain
        - quadrature_weights : the vector containing the quadrature weights of the domain
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - centroid : the vector with values containing the center of mass of the domain.
        - volume : the volume of the domain
        - diameter : the diameter of the domain
        - quadrature_points : the matrix containing the quadrature points of the domain
        - quadrature_weights : the vector containing the quadrature weights of the domain
        """
        self.centroid = centroid
        self.volume = volume
        self.diameter = diameter
        self.quadrature_points = quadrature_points
        self.quadrature_weights = quadrature_weights

    @staticmethod
    def get_domain_barycenter_vector(vertices: Mat) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the barycenter of a domain in the euclidian space.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the vertices of the domain.
        """
        shape_dimension = vertices.shape[1]
        number_of_vertices = vertices.shape[0]
        domain_barycenter = np.zeros((shape_dimension,))
        for vertex in vertices:
            domain_barycenter += vertex
        domain_barycenter = (1.0 / number_of_vertices) * domain_barycenter
        return domain_barycenter
