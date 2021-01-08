import numpy as np
from typing import List
from numpy import ndarray as Mat
from typing import Callable

from core.face import Face
from core.cell import Cell
from bases.basis import Basis
from bases.monomial import ScaledMonomial


class Integration:
    def __init__(self):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Integration class provides methods to compute integration matrices that are then used to define elements.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        All methods are static, so no parameters are needed.
        ================================================================================================================
        Attributes :
        ================================================================================================================
        All methods are static, so no attributes are created.
        """

    @staticmethod
    def get_cell_mass_matrix_in_cell(cell: Cell, cell_basis_0: Basis, cell_basis_1: Basis) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        v_c = cell.diameter
        x_c = cell.centroid
        # --------------------------------------------------------------------------------------------------------------
        cell_mass_matrix_in_cell = np.zeros((cell_basis_0.basis_dimension, cell_basis_1.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        for x_Q_c, w_Q_c in zip(cell.quadrature_points, cell.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            phi_vector_0 = cell_basis_0.get_phi_vector(x_Q_c, x_c, v_c)
            number_of_components = phi_vector_0.shape[0]
            phi_vector_0 = np.resize(phi_vector_0, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            phi_vector_1 = cell_basis_1.get_phi_vector(x_Q_c, x_c, v_c)
            number_of_components = phi_vector_1.shape[0]
            phi_vector_1 = np.resize(phi_vector_1, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            m = w_Q_c * phi_vector_0.T @ phi_vector_1
            cell_mass_matrix_in_cell += m
        return cell_mass_matrix_in_cell

    @staticmethod
    def get_cell_stiffness_matrix_in_cell(cell: Cell, cell_basis_0: Basis, cell_basis_1: Basis, dx: int) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        v_c = cell.diameter
        x_c = cell.centroid
        # --------------------------------------------------------------------------------------------------------------
        cell_stifness_matrix_in_cell = np.zeros((cell_basis_0.basis_dimension, cell_basis_1.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        for x_Q_c, w_Q_c in zip(cell.quadrature_points, cell.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            d_phi_vector_0 = cell_basis_0.get_d_phi_vector(x_Q_c, x_c, v_c, dx)
            number_of_components = d_phi_vector_0.shape[0]
            d_phi_vector_0 = np.resize(d_phi_vector_0, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            d_phi_vector_1 = cell_basis_1.get_d_phi_vector(x_Q_c, x_c, v_c, dx)
            number_of_components = d_phi_vector_1.shape[0]
            d_phi_vector_1 = np.resize(d_phi_vector_1, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            m = w_Q_c * d_phi_vector_0.T @ d_phi_vector_1
            cell_stifness_matrix_in_cell += m
        return cell_stifness_matrix_in_cell

    @staticmethod
    def get_cell_mass_matrix_in_face(cell: Cell, face: Face, cell_basis_0: Basis, cell_basis_1: Basis) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        x_c = cell.centroid
        v_c = cell.diameter
        # --------------------------------------------------------------------------------------------------------------
        cell_mass_matrix_in_face = np.zeros((cell_basis_0.basis_dimension, cell_basis_1.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        for x_Q_f, w_Q_f in zip(face.quadrature_points, face.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            phi_vector_0 = cell_basis_0.get_phi_vector(x_Q_f, x_c, v_c)
            number_of_components = phi_vector_0.shape[0]
            phi_vector_0 = np.resize(phi_vector_0, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            phi_vector_1 = cell_basis_1.get_phi_vector(x_Q_f, x_c, v_c)
            number_of_components = phi_vector_1.shape[0]
            phi_vector_1 = np.resize(phi_vector_1, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            m = w_Q_f * phi_vector_0.T @ phi_vector_1
            cell_mass_matrix_in_face += m
        return cell_mass_matrix_in_face

    @staticmethod
    def get_cell_advection_matrix_in_cell(cell: Cell, cell_basis_0: Basis, cell_basis_1: Basis, dx: int) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        v_c = cell.diameter
        x_c = cell.centroid
        # --------------------------------------------------------------------------------------------------------------
        cell_advection_matrix_in_cell = np.zeros((cell_basis_0.basis_dimension, cell_basis_1.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        for x_Q_c, w_Q_c in zip(cell.quadrature_points, cell.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            phi_vector_0 = cell_basis_0.get_phi_vector(x_Q_c, x_c, v_c)
            number_of_components = phi_vector_0.shape[0]
            phi_vector_0 = np.resize(phi_vector_0, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            d_phi_vector_1 = cell_basis_1.get_d_phi_vector(x_Q_c, x_c, v_c, dx)
            number_of_components = d_phi_vector_1.shape[0]
            d_phi_vector_1 = np.resize(d_phi_vector_1, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            m = w_Q_c * phi_vector_0.T @ d_phi_vector_1
            cell_advection_matrix_in_cell += m
        return cell_advection_matrix_in_cell

    @staticmethod
    def get_cell_advection_matrix_in_face(
        cell: Cell, face: Face, cell_basis_0: Basis, cell_basis_1: Basis, dx: int
    ) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        x_c = cell.centroid
        v_c = cell.diameter
        # --------------------------------------------------------------------------------------------------------------
        cell_advection_matrix_in_face = np.zeros((cell_basis_0.basis_dimension, cell_basis_1.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        for x_Q_f, w_Q_f in zip(face.quadrature_points, face.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            phi_vector_0 = cell_basis_0.get_phi_vector(x_Q_f, x_c, v_c)
            number_of_components = phi_vector_0.shape[0]
            phi_vector_0 = np.resize(phi_vector_0, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            d_phi_vector_1 = cell_basis_1.get_d_phi_vector(x_Q_f, x_c, v_c, dx)
            number_of_components = d_phi_vector_1.shape[0]
            d_phi_vector_1 = np.resize(d_phi_vector_1, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            m = w_Q_f * phi_vector_0.T @ d_phi_vector_1
            cell_advection_matrix_in_face += m
        return cell_advection_matrix_in_face

    @staticmethod
    def get_hybrid_mass_matrix_in_face(
        cell: Cell, face: Face, cell_basis: Basis, face_basis: Basis, face_reference_frame_transformation_matrix: Mat,
    ) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        v_f = face.diameter
        x_f = face.centroid
        v_c = cell.diameter
        x_c = cell.centroid
        # --------------------------------------------------------------------------------------------------------------
        hybrid_mass_matrix_in_face = np.zeros((cell_basis.basis_dimension, face_basis.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        x_f_in_face = Face.get_points_in_face_reference_frame(face.centroid, face_reference_frame_transformation_matrix)
        face_quadrature_points_in_face = Face.get_points_in_face_reference_frame(
            face.quadrature_points, face_reference_frame_transformation_matrix
        )
        # --------------------------------------------------------------------------------------------------------------
        for x_Q, x_Q_in_face, w_Q in zip(
            face.quadrature_points, face_quadrature_points_in_face, face.quadrature_weights
        ):
            # ----------------------------------------------------------------------------------------------------------
            # phi_vector = cell_basis.get_phi_vector(x_Q, x_f, v_f)
            phi_vector = cell_basis.get_phi_vector(x_Q, x_c, v_c)
            number_of_components = phi_vector.shape[0]
            phi_vector = np.resize(phi_vector, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            psi_vector = face_basis.get_phi_vector(x_Q_in_face, x_f_in_face, v_f)
            number_of_components = psi_vector.shape[0]
            psi_vector = np.resize(psi_vector, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            m = w_Q * phi_vector.T @ psi_vector
            hybrid_mass_matrix_in_face += m
        return hybrid_mass_matrix_in_face

    @staticmethod
    def get_hybrid_advection_matrix_in_face(
        cell: Cell,
        face: Face,
        cell_basis: Basis,
        face_basis: Basis,
        face_reference_frame_transformation_matrix: Mat,
        dx: int,
    ) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        v_f = face.diameter
        x_f = face.centroid
        v_c = cell.diameter
        x_c = cell.centroid
        # --------------------------------------------------------------------------------------------------------------
        hybrid_advection_matrix_in_face = np.zeros((cell_basis.basis_dimension, face_basis.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        x_f_in_face = Face.get_points_in_face_reference_frame(face.centroid, face_reference_frame_transformation_matrix)
        face_quadrature_points_in_face = Face.get_points_in_face_reference_frame(
            face.quadrature_points, face_reference_frame_transformation_matrix
        )
        # --------------------------------------------------------------------------------------------------------------
        for x_Q_f, x_Q_in_face, w_Q in zip(
            face.quadrature_points, face_quadrature_points_in_face, face.quadrature_weights
        ):
            # ----------------------------------------------------------------------------------------------------------
            d_phi_vector = cell_basis.get_d_phi_vector(x_Q_f, x_c, v_c, dx)
            number_of_components = d_phi_vector.shape[0]
            d_phi_vector = np.resize(d_phi_vector, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            psi_vector = face_basis.get_phi_vector(x_Q_in_face, x_f_in_face, v_f)
            number_of_components = psi_vector.shape[0]
            psi_vector = np.resize(psi_vector, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            m = w_Q * d_phi_vector.T @ psi_vector
            hybrid_advection_matrix_in_face += m
        return hybrid_advection_matrix_in_face

    @staticmethod
    def get_face_mass_matrix_in_face(
        face: Face, face_basis_0: Basis, face_basis_1: Basis, face_reference_frame_transformation_matrix: Mat
    ) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        v_f = face.diameter
        x_f = face.centroid
        # --------------------------------------------------------------------------------------------------------------
        face_mass_matrix_in_face = np.zeros((face_basis_0.basis_dimension, face_basis_1.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        x_f_in_face = Face.get_points_in_face_reference_frame(face.centroid, face_reference_frame_transformation_matrix)
        face_quadrature_points_in_face = Face.get_points_in_face_reference_frame(
            face.quadrature_points, face_reference_frame_transformation_matrix
        )
        for x_Q_f_in_face, w_Q_f in zip(face_quadrature_points_in_face, face.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            psi_vector_0 = face_basis_0.get_phi_vector(x_Q_f_in_face, x_f_in_face, v_f)
            number_of_components = psi_vector_0.shape[0]
            psi_vector_0 = np.resize(psi_vector_0, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            psi_vector_1 = face_basis_1.get_phi_vector(x_Q_f_in_face, x_f_in_face, v_f)
            number_of_components = psi_vector_1.shape[0]
            psi_vector_1 = np.resize(psi_vector_1, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            m = w_Q_f * psi_vector_0.T @ psi_vector_1
            face_mass_matrix_in_face += m
        return face_mass_matrix_in_face

    @staticmethod
    def get_face_pressure_vector_in_face(
        face: Face, face_basis: Basis, face_reference_frame_transformation_matrix: Mat, pressure: Callable,
    ) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        v_f = face.diameter
        x_f = face.centroid
        # --------------------------------------------------------------------------------------------------------------
        face_pressure_vector_in_face = np.zeros((face_basis.basis_dimension,))
        # --------------------------------------------------------------------------------------------------------------
        x_f_in_face = Face.get_points_in_face_reference_frame(face.centroid, face_reference_frame_transformation_matrix)
        face_quadrature_points_in_face = Face.get_points_in_face_reference_frame(
            face.quadrature_points, face_reference_frame_transformation_matrix
        )
        for x_Q_f_in_face, w_Q_f in zip(face_quadrature_points_in_face, face.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            psi_vector = face_basis.get_phi_vector(x_Q_f_in_face, x_f_in_face, v_f)
            v = w_Q_f * psi_vector * pressure(x_Q_f_in_face)
            face_pressure_vector_in_face += v
        return face_pressure_vector_in_face

    @staticmethod
    def get_cell_load_vector_in_cell(cell: Cell, cell_basis: Basis, load: Callable) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        v_c = cell.diameter
        x_c = cell.centroid
        # --------------------------------------------------------------------------------------------------------------
        cell_load_vector_in_cell = np.zeros((cell_basis.basis_dimension,))
        # --------------------------------------------------------------------------------------------------------------
        for x_Q_c, w_Q_c in zip(cell.quadrature_points, cell.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            phi_vector = cell_basis.get_phi_vector(x_Q_c, x_c, v_c)
            v = w_Q_c * phi_vector * load(x_Q_c)
            cell_load_vector_in_cell += v
        return cell_load_vector_in_cell

    @staticmethod
    def get_cell_integration_vector(cell: Cell, cell_basis: Basis) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        v_c = cell.diameter
        x_c = cell.centroid
        # --------------------------------------------------------------------------------------------------------------
        cell_integration_vector = np.zeros((cell_basis.basis_dimension,))
        # --------------------------------------------------------------------------------------------------------------
        for x_Q_c, w_Q_c in zip(cell.quadrature_points, cell.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            phi_vector = cell_basis.get_phi_vector(x_Q_c, x_c, v_c)
            v = w_Q_c * phi_vector
            cell_integration_vector += v
        return cell_integration_vector
