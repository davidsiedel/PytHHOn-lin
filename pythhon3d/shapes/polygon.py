from core.domain import Domain
from quadratures.dunavant import DunavantRule
from shapes.triangle import Triangle
from numpy import ndarray as Mat

import numpy as np
from scipy.special import binom


class Polygon(Domain):
    def __init__(self, vertices: Mat, polynomial_order: int):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Polygon class inherits from the Domain class to specifiy its attributes when the domain is a polygon.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the vertices coordinates as vectors.
        - polynomial_order : the polynomial order of integration over the polygon.
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - centroid : the vector with values containing the center of mass of the polygon.
        - volume : the volume of the polygon.
        - diameter : the diameter of the polygon.
        - quadrature_points : the matrix containing the quadrature points of the polygon.
        - quadrature_weights : the vector containing the quadrature weights of the polygon.
        """
        if not vertices.shape[0] > 4 and not vertices.shape[1] == 2:
            raise TypeError("The domain dimension do not match that of a polygon")
        else:
            barycenter = Domain.get_domain_barycenter_vector(vertices)
            volume = 0.0
            # volume = Polygon.get_polygon_volume(vertices)
            simplicial_sub_domains = Polygon.get_polygon_simplicial_partition(vertices, barycenter)
            # sub_domains_centroids = []
            quadrature_points, quadrature_weights = [], []
            for simplicial_sub_domain in simplicial_sub_domains:
                # simplex_centroid = Domain.get_domain_barycenter_vector(simplicial_sub_domain)
                simplex_volume = Triangle.get_triangle_volume(simplicial_sub_domain)
                simplex_quadrature_points, simplex_quadrature_weights = DunavantRule.get_triangle_quadrature(
                    simplicial_sub_domain, simplex_volume, polynomial_order
                )
                volume += simplex_volume
                quadrature_points.append(simplex_quadrature_points)
                quadrature_weights.append(simplex_quadrature_weights)
                # sub_domains_centroids.append(simplex_centroid)
            # centroid = np.zeros(2,)
            # for sub_domain_centroid in sub_domains_centroids:
            #     centroid += sub_domain_centroid
            # number_of_vertices = vertices.shape[0]
            # centroid = 1.0 / number_of_vertices * centroid
            centroid = Polygon.get_polygon_centroid(vertices, volume)
            diameter = Polygon.get_polygon_diameter(vertices)
            quadrature_points = np.concatenate(quadrature_points, axis=0)
            quadrature_weights = np.concatenate(quadrature_weights, axis=0)
            super().__init__(centroid, volume, diameter, quadrature_points, quadrature_weights)

    @staticmethod
    def get_polygon_simplicial_partition(vertices: Mat, centroid: Mat) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the partition of a polygon into triangles. Each triangle consists in the barycenter of the polygon, and
        both endpoint of a given face (which is a segment in two dimension).
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the coordinates of the vertices of the polygon as vectors.
        - centroid : the vector containing the coordinates of the barcyenter of the polygon.
        ================================================================================================================
        Returns :
        ================================================================================================================
        - simplicial_sub_domains : the list of matrices containing the vertices of each triangle composing the partition
        of the polygon.
        """
        number_of_vertices = vertices.shape[0]
        simplicial_sub_domains = []
        for i in range(number_of_vertices):
            sub_domain_vertices = [
                vertices[i - 1, :],
                vertices[i, :],
                centroid,
            ]
            simplicial_sub_domains.append(np.array(sub_domain_vertices))
        return simplicial_sub_domains

    @staticmethod
    def get_polygon_volume(vertices: Mat) -> float:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the surface of a polygon.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the coordinates of the vertices of the polygon as vectors.
        ================================================================================================================
        Returns :
        ================================================================================================================
        - polygon_volume : the surface of the polygon.
        """
        number_of_vertices = vertices.shape[0]
        shoe_lace_matrix = []
        for i in range(number_of_vertices):
            lace = vertices[i - 1][0] * vertices[i][1] - vertices[i][0] * vertices[i - 1][1]
            shoe_lace_matrix.append(lace)
        polygon_volume = np.abs(1.0 / 2.0 * np.sum(shoe_lace_matrix))
        # shoe_lace_matrix = []
        # for i in range(number_of_vertices):
        #     edge_matrix = np.array([vertices[i - 1], vertices[i]])
        #     lace = np.linalg.det(edge_matrix.T)
        #     shoe_lace_matrix.append(lace)
        # polygon_volume = np.abs(1.0 / 2.0 * np.sum(shoe_lace_matrix))
        return polygon_volume

    @staticmethod
    def get_polygon_centroid(vertices: Mat, volume: float) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the coordinates of the barcyenter of the polygon from its vertices.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the coordinates of the vertices of the polygon as vectors.
        ================================================================================================================
        Returns :
        ================================================================================================================
        - polygon_centroid : the vector containing the coordinates of the centroid of the polygon.
        """
        number_of_vertices = vertices.shape[0]
        # --------------------------------------------------------------------------------------------------------------
        # x coordinates
        # --------------------------------------------------------------------------------------------------------------
        centroid_matrix_x = []
        for i in range(number_of_vertices):
            centroid_matrix_x.append(
                (vertices[i - 1][0] + vertices[i][0])
                * (vertices[i - 1][0] * vertices[i][1] - vertices[i][0] * vertices[i - 1][1])
            )
        polygon_centroid_x = 1.0 / (6.0 * volume) * np.sum(centroid_matrix_x)
        # --------------------------------------------------------------------------------------------------------------
        # y coordinates
        # --------------------------------------------------------------------------------------------------------------
        centroid_matrix_y = []
        for i in range(number_of_vertices):
            centroid_matrix_y.append(
                (vertices[i - 1][1] + vertices[i][1])
                * (vertices[i - 1][0] * vertices[i][1] - vertices[i][0] * vertices[i - 1][1])
            )
        polygon_centroid_y = 1.0 / (6.0 * volume) * np.sum(centroid_matrix_y)
        # --------------------------------------------------------------------------------------------------------------
        # centroid
        # --------------------------------------------------------------------------------------------------------------
        polygon_centroid = np.array([polygon_centroid_x, polygon_centroid_y])
        return polygon_centroid

    @staticmethod
    def get_polygon_diameter(vertices: Mat) -> float:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the diamter of a polygon.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the coordinates of the vertices of the polygon as vectors.
        ================================================================================================================
        Returns :
        ================================================================================================================
        - polygon_diameter : the diameter of the polygon.
        """
        number_of_vertices = vertices.shape[0]
        number_of_combinations = binom(number_of_vertices, 2)
        # --------------------------------------------------------------------------------------------------------------
        combinations_count = 0
        lengths = []
        for i, v0 in enumerate(vertices):
            for j, v1 in enumerate(vertices):
                # if not i == j and not permutation_count == number_of_combinations:
                if not i == j:
                    e = v1 - v0
                    lengths.append(np.sqrt((e[1] + e[0]) ** 2))
        polygon_diameter = max(lengths)
        return polygon_diameter
