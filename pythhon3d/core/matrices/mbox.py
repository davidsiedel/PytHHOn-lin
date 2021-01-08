import numpy as np
from numpy import ndarray as Mat
from typing import List
from typing import Callable

from core.cell import Cell
from core.face import Face
from core.matrices.cell import Mcell
from core.matrices.face import Mface
from core.unknown import Unknown
from core.integration import Integration
from bases.basis import Basis


class Mbox:
    def __init__(
        cell: Cell,
        faces: List[Face],
        face_basis_k: Basis,
        cell_basis_l: Basis,
        cell_basis_k: Basis,
        cell_basis_k1: Basis,
        unknown: Unknown,
    ):
        (
            self.mat_T_phi_k_phi_k_inv,
            self.mat_T_phi_k_grad_phi_l_list,
            self.mat_T_grad_phi_k1_grad_phi_k1_inv_list,
            self.mat_T_grad_phi_k1_phi_l_list,
            self.proj_T_T_k1_k,
        ) = self.compute_cell_matrices(cell, cell_basis_l, cell_basis_k, cell_basis_k1, unknown)
        # --------------------------------------------------------------------------------------------------------------
        self.mat_F_psi_k_psi_k_inv_list = self.compute_faces_matrices(faces, face_basis_k)
        # --------------------------------------------------------------------------------------------------------------
        (
            self.mat_F_phi_k_phi_l_list,
            self.mat_F_phi_k_psi_k_list,
            self.mat_F_grad_phi_k1_psi_k_list,
            self.mat_F_grad_phi_k1_phi_l_list,
            self.mat_F_psi_k_phi_k1_list,
        ) = self.compute_hybrid_matrices(cell, faces, cell_basis_l, cell_basis_k, cell_basis_k1, face_basis_k, unknown)

    def compute_hybrid_matrices(
        self,
        cell: Cell,
        faces: List[Face],
        cell_basis_l: Basis,
        cell_basis_k: Basis,
        cell_basis_k1: Basis,
        face_basis_k: Basis,
        unknown: Unknown,
    ):
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
        # ==============================================================================================================
        # GRADIENT OPERATOR / STABILIZATION OPERATOR
        # ==============================================================================================================
        mat_F_phi_k_phi_l_list = []
        mat_F_phi_k_psi_k_list = []
        for mface in mfaces:
            mat_F_phi_k_phi_l = Integration.get_cell_mass_matrix_in_face(
                mcell.cell, mface.face, cell_basis_k, cell_basis_l
            )
            mat_F_phi_k_phi_l_list.append(mat_F_phi_k_phi_l_list)
            # ----------------------------------------------------------------------------------------------------------
            mat_F_phi_k_psi_k = Integration.get_hybrid_mass_matrix_in_face(
                mcell.cell, mface.face, cell_basis_k, face_basis_k, mface.face.reference_frame_transformation_matrix
            )
            mat_F_phi_k_psi_k_list.append(mat_F_phi_k_psi_k)
        # ==============================================================================================================
        # RECONSTRUCTION OPERATOR
        # ==============================================================================================================
        mat_F_grad_phi_k1_psi_k_list = []
        mat_F_grad_phi_k1_phi_l_list = []
        for mface in mfaces:
            mat_F_grad_phi_k1_psi_k_j_list = []
            mat_F_grad_phi_k1_phi_l_j_list = []
            for j in range(len(unknown.problem_dimension)):
                # ------------------------------------------------------------------------------------------------------
                mat_F_grad_phi_k1_psi_k_j = Integration.get_hybrid_advection_matrix_in_face(
                    mcell.cell,
                    mface.face,
                    cell_basis_k1,
                    face_basis_k,
                    mface.face.reference_frame_transformation_matrix,
                    j,
                )
                mat_F_grad_phi_k1_psi_k_j_list.append(mat_F_grad_phi_k1_psi_k_j)
                # ------------------------------------------------------------------------------------------------------
                mat_F_phi_l_grad_phi_k1_j = Integration.get_cell_advection_matrix_in_face(
                    mcell.cell, mface.face, cell_basis_l, cell_basis_k1, j
                )
                # ------------------------------------------------------------------------------------------------------
                mat_F_grad_phi_k1_phi_l_j = mat_F_phi_l_grad_phi_k1_j.T
                mat_F_grad_phi_k1_phi_l_j_list.append(mat_F_grad_phi_k1_phi_l_j)
            # ----------------------------------------------------------------------------------------------------------
            mat_F_grad_phi_k1_psi_k_list.append(mat_F_grad_phi_k1_psi_k_j_list)
            mat_F_grad_phi_k1_phi_l_list.append(mat_F_grad_phi_k1_phi_l_j_list)
        # ==============================================================================================================
        # STABILIZATION OPERATOR
        # ==============================================================================================================
        mat_F_psi_k_phi_k1_list = []
        for mface in mfaces:
            mat_F_phi_k1_psi_k = Integration.get_hybrid_mass_matrix_in_face(
                mcell.cell, mface.face, cell_basis_k1, face_basis_k, mface.face.reference_frame_transformation_matrix
            )
            mat_F_psi_k_phi_k1_list.append(mat_F_phi_k1_psi_k)
        return (
            mat_F_phi_k_phi_l_list,
            mat_F_phi_k_psi_k_list,
            mat_F_grad_phi_k1_psi_k_list,
            mat_F_grad_phi_k1_phi_l_list,
            mat_F_psi_k_phi_k1_list,
        )

    def compute_faces_matrices(self, faces: List[Face], face_basis_k: Basis):
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
        mat_F_psi_k_psi_k_inv_list = []
        for face in faces:
            mat_F_psi_k_psi_k = Integration.get_face_mass_matrix_in_face(
                face, face_basis_k, face_basis_k, face.reference_frame_transformation_matrix
            )
            mat_F_psi_k_psi_k_inv = np.linalg.inv(mat_F_psi_k_psi_k)
            mat_F_psi_k_psi_k_inv_list.append(mat_F_psi_k_psi_k_inv)
        return mat_F_psi_k_psi_k_inv_list

    def compute_cell_matrices(
        self, cell: Cell, cell_basis_l: Basis, cell_basis_k: Basis, cell_basis_k1: Basis, unknown: Unknown
    ):
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
        # ==============================================================================================================
        # GRADIENT OPERATOR
        # ==============================================================================================================
        # left-hand side
        # --------------------------------------------------------------------------------------------------------------
        mat_T_phi_k_phi_k = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis_k, cell_basis_k)
        mat_T_phi_k_phi_k_inv = np.linalg.inv(mat_T_phi_k_phi_k)
        # --------------------------------------------------------------------------------------------------------------
        # right-hand side
        # --------------------------------------------------------------------------------------------------------------
        mat_T_phi_k_grad_phi_l_list = []
        for j in range(unknown.problem_dimension):
            mat_T_phi_k_grad_phi_l = Integration.get_cell_advection_matrix_in_cell(cell, cell_basis_k, cell_basis_l, j)
            mat_T_phi_k_grad_phi_l_list.append(mat_T_phi_k_grad_phi_l)
        # ==============================================================================================================
        # RECONSTRUCTION OPERATOR
        # ==============================================================================================================
        # left-hand side
        # --------------------------------------------------------------------------------------------------------------
        mat_T_grad_phi_k1_grad_phi_k1_inv_list = []
        for j in range(unknown.problem_dimension):
            mat_T_grad_phi_k1_grad_phi_k1_j = Integration.get_cell_stiffness_matrix_in_cell(
                cell, cell_basis_k1, cell_basis_k1, j
            )
            mat_T_grad_phi_k1_grad_phi_k1_inv_list.append(mat_T_grad_phi_k1_grad_phi_k1_j)
        # --------------------------------------------------------------------------------------------------------------
        # right-hand side
        # --------------------------------------------------------------------------------------------------------------
        mat_T_grad_phi_k1_phi_l_list = []
        for j in range(unknown.problem_dimension):
            mat_T_phi_l_grad_phi_k1_j = Integration.get_cell_advection_matrix_in_cell(
                cell, cell_basis_l, cell_basis_k1, j
            )
            mat_T_grad_phi_k1_phi_l_j = mat_T_phi_l_grad_phi_k1_j.T
            mat_T_grad_phi_k1_phi_l_list.append(mat_T_grad_phi_k1_phi_l_j)
        # ==============================================================================================================
        # STABILIZATION OPERATOR
        # ==============================================================================================================
        mat_T_phi_k_phi_k1 = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis_k, cell_basis_k1)
        proj_T_T_k1_k = mat_T_phi_k_phi_k_inv @ mat_T_phi_k_phi_k1
        # ==============================================================================================================
        # PROJECTION OPERATOR
        # ==============================================================================================================
        mat_T_phi_l_phi_l = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis_l, cell_basis_l)
        mat_T_phi_l_phi_l_inv = np.linalg.inv(mat_T_phi_l_phi_l)
        return (
            mat_T_phi_k_phi_k_inv,
            mat_T_phi_k_grad_phi_l_list,
            mat_T_grad_phi_k1_grad_phi_k1_inv_list,
            mat_T_grad_phi_k1_phi_l_list,
            proj_T_T_k1_k,
        )
