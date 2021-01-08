from core.domain import Domain
from quadratures.dunavant import DunavantRule
from shapes.tetrahedron import Tetrahedron
from shapes.polygon import Polygon
from numpy import ndarray as Mat

import numpy as np


class Polyhedron(Domain):
    def __init__(self, vertices: Mat, connectivity_matrix: Mat, polynomial_order: int):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Polyhedron class inherits from the Domain class to specifiy its attributes when the domain is a polygon.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the vertices coordinates as vectors.
        - polynomial_order : the polynomial order of integration over the polyhedron.
        - connectivity_matrix : the matrix that specifies the connection between the vertices of the polyhedron and its 
        faces, as a list of indices corresponding to the vertex indices for each face. The number of rows is the number of faces, and for each row, the number of columns is the number of vertices composing the face.
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - centroid : the vector with values containing the center of mass of the polyhedron.
        - volume : the volume of the polyhedron.
        - diameter : the diameter of the polyhedron.
        - quadrature_points : the matrix containing the quadrature points of the polyhedron.
        - quadrature_weights : the vector containing the quadrature weights of the polyhedron.
        """
        if not vertices.shape[0] > 4 and not vertices.shape[1] == 2:
            raise TypeError("The domain dimensions do not match that of a polygon")
        else:
            barycenter = Domain.get_domain_barycenter_vector(vertices)
            simplicial_sub_domains = Polyhedron.get_polyhedron_simplicial_partition(
                vertices, connectivity_matrix, barycenter
            )
            # print(simplicial_sub_domains)
            volume = 0.0
            diameter = None
            quadrature_points, quadrature_weights = [], []
            for simplicial_sub_domain in simplicial_sub_domains:
                simplex_volume = Tetrahedron.get_tetrahedron_volume(simplicial_sub_domain)
                simplex_quadrature_points, simplex_quadrature_weights = DunavantRule.get_tetrahedron_quadrature(
                    simplicial_sub_domain, simplex_volume, polynomial_order
                )
                volume += simplex_volume
                quadrature_points.append(simplex_quadrature_points)
                quadrature_weights.append(simplex_quadrature_weights)
            quadrature_points = np.concatenate(quadrature_points, axis=0)
            quadrature_weights = np.concatenate(quadrature_weights, axis=0)
            super().__init__(barycenter, volume, diameter, quadrature_points, quadrature_weights)

    @staticmethod
    def get_polyhedron_simplicial_partition(vertices: Mat, connectivity_matrix: Mat, barycenter: Mat) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the partition of a polyhedron into tetrahedra. Each tetrahedron consists in the barycenter of the polygon, and a triangle that composing the simplicial partition of a face.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the matrix containing the vertices coordinates as vectors.
        - connectivity_matrix : the matrix that specifies the connection between the vertices of the polyhedron and its 
        faces, as a list of indices corresponding to the vertex indices for each face. The number of rows is the number of faces, and for each row, the number of columns is the number of vertices composing the face.
        - polynomial_order : the polynomial order of integration over the polyhedron.
        ================================================================================================================
        Returns :
        ================================================================================================================
        - simplicial_sub_domains : the list of matrices containing the vertices of each tetrahedron composing the
        partition of the polyhedron.
        """
        simplicial_sub_domains = []
        for face_vertices_indexes in connectivity_matrix:
            face_vertices = vertices[face_vertices_indexes]
            if face_vertices.shape[0] > 3:
                face_barycenter = Domain.get_domain_barycenter_vector(face_vertices)
                sub_faces = Polygon.get_polygon_simplicial_partition(face_vertices, face_barycenter)
                for sub_face in sub_faces:
                    b = np.resize(barycenter, (1, 3))
                    tetra = np.concatenate((sub_face, b), axis=0)
                    simplicial_sub_domains.append(tetra)
            else:
                b = np.resize(barycenter, (1, 3))
                tetra = np.concatenate((face_vertices, b), axis=0)
                simplicial_sub_domains.append(tetra)
        return simplicial_sub_domains
