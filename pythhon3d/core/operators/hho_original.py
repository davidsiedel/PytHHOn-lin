from core.face import Face
from core.cell import Cell
from core.integration import Integration
from core.operators.operator import Operator
from core.unknown import Unknown
from bases.basis import Basis
from bases.monomial import ScaledMonomial

import numpy as np
from typing import List
from numpy import ndarray as Mat


class HHOr(Operator):
    def __init__(
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
        Class :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        
        ================================================================================================================
        Attributes :
        ================================================================================================================
        
        """
        local_stabilization_operator = self.get_stabilization_operator(
            cell, faces, cell_basis_l, cell_basis_k, cell_basis_k1, face_basis_k, unknown
        )
        local_gradient_operator = self.get_local_gradient_operator(
            cell, faces, cell_basis_l, cell_basis_k, cell_basis_k1, face_basis_k, unknown
        )
        local_mass_operator = self.get_local_mass_operator(
            cell, faces, cell_basis_l, cell_basis_k, cell_basis_k1, face_basis_k, unknown
        )
        super().__init__(local_gradient_operator, local_stabilization_operator, local_mass_operator)

    def get_local_gradient_operator(
        self,
        cell: Cell,
        faces: List[Face],
        cell_basis_l: Basis,
        cell_basis_k: Basis,
        cell_basis_k1: Basis,
        face_basis_k: Basis,
        unknown: Unknown,
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
        # --------------------------------------------------------------------------------------------------------------
        # Getting the local problem size depending on the problem dimension and the field dimension
        # --------------------------------------------------------------------------------------------------------------
        local_problem_size = self.get_local_problem_size(faces, cell_basis_l, face_basis_k, unknown)
        # --------------------------------------------------------------------------------------------------------------
        # Initializing the local gradient operator matrix
        # --------------------------------------------------------------------------------------------------------------
        local_gradient_operator = np.zeros((len(unknown.indices) * cell_basis_k.basis_dimension, local_problem_size))
        # --------------------------------------------------------------------------------------------------------------
        m_cell_phi_k_phi_k = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis_k, cell_basis_k)
        m_cell_phi_k_phi_k_inv = np.linalg.inv(m_cell_phi_k_phi_k)
        # --------------------------------------------------------------------------------------------------------------
        derivative_directions = range(unknown.problem_dimension)
        directions = range(unknown.field_dimension)
        for j in derivative_directions:
            # ----------------------------------------------------------------------------------------------------------
            # Compting matrices 1
            # ----------------------------------------------------------------------------------------------------------
            m_cell_phi_k_grad_phi_l = Integration.get_cell_advection_matrix_in_cell(cell, cell_basis_k, cell_basis_l, j)
            for f, face in enumerate(faces):
                passmat = Operator.get_face_passmat(cell, face)
                normal_vector_component = passmat[-1, j]
                # m_face_phi_l_psi_k = Integration.get_hybrid_mass_matrix_in_face(
                #     cell, face, cell_basis_l, face_basis_k, passmat
                # )
                m_face_phi_k_psi_k = Integration.get_hybrid_mass_matrix_in_face(
                    cell, face, cell_basis_k, face_basis_k, passmat
                )
                m_face_phi_k_phi_l = Integration.get_cell_mass_matrix_in_face(cell, face, cell_basis_k, cell_basis_l)
                for i in directions:
                    # --------------------------------------------------------------------------------------------------
                    # bkbjkj
                    # --------------------------------------------------------------------------------------------------
                    line = self.get_line_from_indices(i, j, unknown)
                    r0 = line * cell_basis_k.basis_dimension
                    r1 = (line + 1) * cell_basis_k.basis_dimension
                    # --------------------------------------------------------------------------------------------------
                    c0_T = i * cell_basis_l.basis_dimension
                    c1_T = (i + 1) * cell_basis_l.basis_dimension
                    # --------------------------------------------------------------------------------------------------
                    # Getting the indices corresponding to the face_index-th face for the field on the ith axis
                    # --------------------------------------------------------------------------------------------------
                    c0_F = (
                        unknown.field_dimension * cell_basis_l.basis_dimension
                        + f * unknown.field_dimension * face_basis_k.basis_dimension
                        + i * face_basis_k.basis_dimension
                    )
                    c1_F = (
                        unknown.field_dimension * cell_basis_l.basis_dimension
                        + f * unknown.field_dimension * face_basis_k.basis_dimension
                        + (i + 1) * face_basis_k.basis_dimension
                    )
                    # --------------------------------------------------------------------------------------------------
                    # grad
                    # --------------------------------------------------------------------------------------------------
                    # local_gradient_operator[r0:r1, c0_T:c1_T] += m_cell_phi_k_grad_phi_l
                    local_gradient_operator[r0:r1, c0_F:c1_F] += m_face_phi_k_psi_k * normal_vector_component
                    local_gradient_operator[r0:r1, c0_T:c1_T] -= m_face_phi_k_phi_l * normal_vector_component
            for i in directions:
                line = self.get_line_from_indices(i, j, unknown)
                r0 = line * cell_basis_k.basis_dimension
                r1 = (line + 1) * cell_basis_k.basis_dimension
                # ------------------------------------------------------------------------------------------------------
                c0_T = i * cell_basis_l.basis_dimension
                c1_T = (i + 1) * cell_basis_l.basis_dimension
                local_gradient_operator[r0:r1, c0_T:c1_T] += m_cell_phi_k_grad_phi_l
                local_gradient_operator[r0:r1, :] = m_cell_phi_k_phi_k_inv @ local_gradient_operator[r0:r1, :]
        return local_gradient_operator

    def get_local_reconstruction_operator(
        self,
        cell: Cell,
        faces: List[Face],
        cell_basis_l: Basis,
        cell_basis_k: Basis,
        cell_basis_k1: Basis,
        face_basis_k: Basis,
        unknown: Unknown,
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
        # --------------------------------------------------------------------------------------------------------------
        # Getting the local problem size depending on the problem dimension and the field dimension
        # --------------------------------------------------------------------------------------------------------------
        local_problem_size = self.get_local_problem_size(faces, cell_basis_l, face_basis_k, unknown)
        # --------------------------------------------------------------------------------------------------------------
        # Initializing the local gradient operator matrix
        # --------------------------------------------------------------------------------------------------------------
        local_reconstruction_operator = np.zeros(
            ((unknown.field_dimension) * cell_basis_k1.basis_dimension, local_problem_size)
        )
        # --------------------------------------------------------------------------------------------------------------
        derivative_directions = range(unknown.problem_dimension)
        directions = range(unknown.field_dimension)
        m_cell_grad_phi_k1_grad_phi_k1_sum = np.zeros((cell_basis_k1.basis_dimension, cell_basis_k1.basis_dimension))
        m_cell_grad_phi_k1_grad_phi_l_sum = np.zeros((cell_basis_k1.basis_dimension, cell_basis_l.basis_dimension))
        for j in derivative_directions:
            m_cell_grad_phi_k1_grad_phi_k1 = Integration.get_cell_stiffness_matrix_in_cell(
                cell, cell_basis_k1, cell_basis_k1, j
            )
            m_cell_grad_phi_k1_grad_phi_k1_sum += m_cell_grad_phi_k1_grad_phi_k1
            m_cell_grad_phi_k1_grad_phi_l = Integration.get_cell_stiffness_matrix_in_cell(
                cell, cell_basis_k1, cell_basis_l, j
            )
            m_cell_grad_phi_k1_grad_phi_l_sum += m_cell_grad_phi_k1_grad_phi_l
        for f, face in enumerate(faces):
            passmat = Operator.get_face_passmat(cell, face)
            normal_vector_component = passmat[-1, j]
            m_face_grad_phi_k1_phi_l_sum = np.zeros((cell_basis_k1.basis_dimension, cell_basis_l.basis_dimension))
            m_face_grad_phi_k1_psi_k_sum = np.zeros((cell_basis_k1.basis_dimension, face_basis_k.basis_dimension))
            for j in derivative_directions:
                m_face_grad_phi_k1_psi_k = Integration.get_hybrid_advection_matrix_in_face(
                    cell, face, cell_basis_k1, face_basis_k, passmat, j
                )
                m_face_grad_phi_k1_psi_k_sum += m_face_grad_phi_k1_psi_k
                m_face_phi_l_grad_phi_k1 = Integration.get_cell_advection_matrix_in_face(
                    cell, face, cell_basis_l, cell_basis_k1, j
                )
                m_face_grad_phi_k1_phi_l = m_face_phi_l_grad_phi_k1.T
                m_face_grad_phi_k1_phi_l_sum += m_face_grad_phi_k1_phi_l
            for i in directions:
                # --------------------------------------------------------------------------------------------------
                # bkbjkj
                # --------------------------------------------------------------------------------------------------
                r0 = i * cell_basis_k1.basis_dimension
                r1 = (i + 1) * cell_basis_k1.basis_dimension
                # --------------------------------------------------------------------------------------------------
                # Getting the indices corresponding to the face_index-th face for the field on the ith axis
                # --------------------------------------------------------------------------------------------------
                c0_F = (
                    unknown.field_dimension * cell_basis_l.basis_dimension
                    + f * unknown.field_dimension * face_basis_k.basis_dimension
                    + i * face_basis_k.basis_dimension
                )
                c1_F = (
                    unknown.field_dimension * cell_basis_l.basis_dimension
                    + f * unknown.field_dimension * face_basis_k.basis_dimension
                    + (i + 1) * face_basis_k.basis_dimension
                )
                c0_T = i * cell_basis_l.basis_dimension
                c1_T = (i + 1) * cell_basis_l.basis_dimension
                # --------------------------------------------------------------------------------------------------
                # rec
                # --------------------------------------------------------------------------------------------------
                local_reconstruction_operator[r0:r1, c0_F:c1_F] += (
                    m_face_grad_phi_k1_psi_k_sum * normal_vector_component
                )
                local_reconstruction_operator[r0:r1, c0_T:c1_T] -= (
                    m_face_grad_phi_k1_phi_l_sum * normal_vector_component
                )
        for i in directions:
            r0 = i * cell_basis_k1.basis_dimension
            r1 = (i + 1) * cell_basis_k1.basis_dimension
            m_cell_grad_phi_k1_grad_phi_k1_sum_inv = np.linalg.inv(m_cell_grad_phi_k1_grad_phi_k1_sum[1:, 1:])
            local_reconstruction_operator[r0 + 1 : r1, :] = (
                m_cell_grad_phi_k1_grad_phi_k1_sum_inv @ local_reconstruction_operator[r0 + 1 : r1, :]
            )
            # ------------------------------------------------------------------------------------------------------
            cell_integration_vector_k1 = Integration.get_cell_integration_vector(cell, cell_basis_k1)
            cell_integration_vector_l = Integration.get_cell_integration_vector(cell, cell_basis_l)
            # ------------------------------------------------------------------------------------------------------
            m_mat = np.zeros((cell_basis_l.basis_dimension, local_problem_size))
            c0_m = i * cell_basis_l.basis_dimension
            c1_m = (i + 1) * cell_basis_l.basis_dimension
            m_mat[:, c0_m:c1_m] += np.eye(cell_basis_l.basis_dimension)
            # ------------------------------------------------------------------------------------------------------
            a = (
                cell_integration_vector_l @ m_mat
                - cell_integration_vector_k1[1:] @ local_reconstruction_operator[r0 + 1 : r1, :]
            )
            a = (1.0 / cell_integration_vector_k1[0]) * a
            local_reconstruction_operator[r0 : r0 + 1, :] = a
        # for i in directions:
        #     r0 = i * cell_basis_k1.basis_dimension
        #     r1 = (i + 1) * cell_basis_k1.basis_dimension
        #     c0_T = i * cell_basis_l.basis_dimension
        #     c1_T = (i + 1) * cell_basis_l.basis_dimension
        #     #
        #     local_reconstruction_operator[r0:r1, c0_T:c1_T] += m_cell_grad_phi_k1_grad_phi_l_sum
        #     local_reconstruction_operator[r0:r1, :] = local_reconstruction_operator[r0:r1, :]

        # for j in derivative_directions:
        #     # ------------------------------------------------------------------------------------------------------
        #     # Compting matrices 1
        #     # ------------------------------------------------------------------------------------------------------
        #     m_cell_grad_phi_k1_grad_phi_k1 = Integration.get_cell_stiffness_matrix_in_cell(
        #         cell, cell_basis_k1, cell_basis_k1, j
        #     )
        #     m_cell_grad_phi_k1_grad_phi_k1_sum += m_cell_grad_phi_k1_grad_phi_k1
        #     m_cell_grad_phi_k1_grad_phi_l = Integration.get_cell_stiffness_matrix_in_cell(
        #         cell, cell_basis_k1, cell_basis_l, j
        #     )
        #     for f, face in enumerate(faces):
        #         passmat = Operator.get_face_passmat(cell, face)
        #         normal_vector_component = passmat[-1, j]
        #         m_face_grad_phi_k1_psi_k
        #         m_face_grad_phi_k1_psi_k = Integration.get_hybrid_advection_matrix_in_face(
        #             cell, face, cell_basis_k1, face_basis_k, passmat, j
        #         )
        #         # m_face_grad_phi_k1_phi_l = Integration.get_cell_advection_matrix_in_face(
        #         #     cell, face, cell_basis_k1, cell_basis_l, j
        #         # )
        #         m_face_phi_l_grad_phi_k1 = Integration.get_cell_advection_matrix_in_face(
        #             cell, face, cell_basis_l, cell_basis_k1, j
        #         )
        #         m_face_grad_phi_k1_phi_l = m_face_phi_l_grad_phi_k1.T
        #         for i in directions:
        #             # --------------------------------------------------------------------------------------------------
        #             # bkbjkj
        #             # --------------------------------------------------------------------------------------------------
        #             r0 = i * cell_basis_k1.basis_dimension
        #             r1 = (i + 1) * cell_basis_k1.basis_dimension
        #             # --------------------------------------------------------------------------------------------------
        #             c0_T = i * cell_basis_l.basis_dimension
        #             c1_T = (i + 1) * cell_basis_l.basis_dimension
        #             # --------------------------------------------------------------------------------------------------
        #             # Getting the indices corresponding to the face_index-th face for the field on the ith axis
        #             # --------------------------------------------------------------------------------------------------
        #             c0_F = (
        #                 unknown.field_dimension * cell_basis_l.basis_dimension
        #                 + f * unknown.field_dimension * face_basis_k.basis_dimension
        #                 + i * face_basis_k.basis_dimension
        #             )
        #             c1_F = (
        #                 unknown.field_dimension * cell_basis_l.basis_dimension
        #                 + f * unknown.field_dimension * face_basis_k.basis_dimension
        #                 + (i + 1) * face_basis_k.basis_dimension
        #             )
        #             # --------------------------------------------------------------------------------------------------
        #             # rec
        #             # --------------------------------------------------------------------------------------------------
        #             local_reconstruction_operator[r0:r1, c0_F:c1_F] += (
        #                 m_face_grad_phi_k1_psi_k * normal_vector_component
        #             )
        #             local_reconstruction_operator[r0:r1, c0_T:c1_T] -= (
        #                 m_face_grad_phi_k1_phi_l * normal_vector_component
        #             )
        #     for i in directions:
        #         r0 = i * cell_basis_k1.basis_dimension
        #         r1 = (i + 1) * cell_basis_k1.basis_dimension
        #         # ------------------------------------------------------------------------------------------------------
        #         c0_T = i * cell_basis_l.basis_dimension
        #         c1_T = (i + 1) * cell_basis_l.basis_dimension
        #         local_reconstruction_operator[r0:r1, c0_T:c1_T] += m_cell_grad_phi_k1_grad_phi_l
        # # --------------------------------------------------------------------------------------------------------------
        # for i in directions:
        #     # a_sum = np.zeros((local_problem_size,))
        #     # for i_sum in directions:
        #     r0 = i * cell_basis_k1.basis_dimension
        #     r1 = (i + 1) * cell_basis_k1.basis_dimension
        #     # print("grad_matrx : \n{}".format(m_cell_grad_phi_k1_grad_phi_k1_sum[1:, 1:]))
        #     m_cell_grad_phi_k1_grad_phi_k1_sum_inv = np.linalg.inv(m_cell_grad_phi_k1_grad_phi_k1_sum[1:, 1:])
        #     local_reconstruction_operator[r0 + 1 : r1, :] = (
        #         m_cell_grad_phi_k1_grad_phi_k1_sum_inv @ local_reconstruction_operator[r0 + 1 : r1, :]
        #     )
        #     # ------------------------------------------------------------------------------------------------------
        #     cell_integration_vector_k1 = Integration.get_cell_integration_vector(cell, cell_basis_k1)
        #     cell_integration_vector_l = Integration.get_cell_integration_vector(cell, cell_basis_l)
        #     # ------------------------------------------------------------------------------------------------------
        #     m_mat = np.zeros((cell_basis_l.basis_dimension, local_problem_size))
        #     c0_m = i * cell_basis_l.basis_dimension
        #     c1_m = (i + 1) * cell_basis_l.basis_dimension
        #     m_mat[:, c0_m:c1_m] += np.eye(cell_basis_l.basis_dimension)
        #     # ------------------------------------------------------------------------------------------------------
        #     a = (
        #         cell_integration_vector_l @ m_mat
        #         - cell_integration_vector_k1[1:] @ local_reconstruction_operator[r0 + 1 : r1, :]
        #     )
        #     a = (1.0 / cell_integration_vector_k1[0]) * a
        #     local_reconstruction_operator[r0 : r0 + 1, :] = a
        return local_reconstruction_operator

    def get_local_mass_operator(
        self,
        cell: Cell,
        faces: List[Face],
        cell_basis_l: Basis,
        cell_basis_k: Basis,
        cell_basis_k1: Basis,
        face_basis_k: Basis,
        unknown: Unknown,
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
        # local_problem_size = self.get_local_problem_size(faces, cell_basis_l, face_basis_k, unknown)
        # local_mass_operator = np.zeros((cell_basis_l.basis_dimension * unknown.field_dimension, local_problem_size))
        # m_cell_phi_l_phi_l = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis_l, cell_basis_l)
        # directions = range(unknown.field_dimension)
        # for i in directions:
        #     r0 = i * cell_basis_l.basis_dimension
        #     r1 = (i + 1) * cell_basis_l.basis_dimension
        #     local_mass_operator[r0:r1, r0:r1] += m_cell_phi_l_phi_l
        # return local_mass_operator
        local_problem_size = self.get_local_problem_size(faces, cell_basis_l, face_basis_k, unknown)
        # local_mass_operator = np.zeros((local_problem_size, local_problem_size))
        local_mass_operator = np.zeros((unknown.field_dimension * cell_basis_l.basis_dimension, local_problem_size))
        # m_cell_phi_l_phi_l = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis_l, cell_basis_l)
        m_cell_id = np.eye(cell_basis_l.basis_dimension)
        directions = range(unknown.field_dimension)
        for i in directions:
            r0 = i * cell_basis_l.basis_dimension
            r1 = (i + 1) * cell_basis_l.basis_dimension
            # local_mass_operator[r0:r1, r0:r1] += m_cell_phi_l_phi_l
            local_mass_operator[r0:r1, r0:r1] += m_cell_id
        return local_mass_operator

    def get_stabilization_operator(
        self,
        cell: Cell,
        faces: List[Face],
        cell_basis_l: Basis,
        cell_basis_k: Basis,
        cell_basis_k1: Basis,
        face_basis_k: Basis,
        unknown: Unknown,
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
        # --------------------------------------------------------------------------------------------------------------
        # rec
        # --------------------------------------------------------------------------------------------------------------
        local_reconstruction_operator = self.get_local_reconstruction_operator(
            cell, faces, cell_basis_l, cell_basis_k, cell_basis_k1, face_basis_k, unknown
        )
        # --------------------------------------------------------------------------------------------------------------
        # Initializing the local stabilization form matrix
        # --------------------------------------------------------------------------------------------------------------
        local_problem_size = self.get_local_problem_size(faces, cell_basis_l, face_basis_k, unknown)
        local_stabilization_operator = np.zeros((local_problem_size, local_problem_size))
        derivative_directions = range(unknown.problem_dimension)
        directions = range(unknown.field_dimension)
        for i in directions:
            # # ------------------------------------------------------------------------------------------------------
            # r0_r = i * cell_basis_k1.basis_dimension
            # r1_r = (i + 1) * cell_basis_k1.basis_dimension
            # ------------------------------------------------------------------------------------------------------
            c0_c = i * cell_basis_l.basis_dimension
            c1_c = (i + 1) * cell_basis_l.basis_dimension
            for f, face in enumerate(faces):
                passmat = Operator.get_face_passmat(cell, face)
                # normal_vector_component = passmat[-1, j]
                # --------------------------------------------------------------------------------------------------
                # Getting the indices corresponding to the face_indexth face for the field on the ith axis
                # --------------------------------------------------------------------------------------------------
                c0_f = (
                    unknown.field_dimension * cell_basis_l.basis_dimension
                    + f * unknown.field_dimension * face_basis_k.basis_dimension
                    + i * face_basis_k.basis_dimension
                )
                c1_f = (
                    unknown.field_dimension * cell_basis_l.basis_dimension
                    + f * unknown.field_dimension * face_basis_k.basis_dimension
                    + (i + 1) * face_basis_k.basis_dimension
                )
                # ------------------------------------------------------------------------------------------------------
                # Compting matrices 1
                # ------------------------------------------------------------------------------------------------------
                m_face_id = np.eye(face_basis_k.basis_dimension)
                m_face_psi_k_psi_k = Integration.get_face_mass_matrix_in_face(face, face_basis_k, face_basis_k, passmat)
                m_face_psi_k_psi_k_inv = np.linalg.inv(m_face_psi_k_psi_k)
                # ------------------------------------------------------------------------------------------------------
                m_cell_phi_k_phi_k = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis_k, cell_basis_k)
                m_cell_phi_k_phi_k_inv = np.linalg.inv(m_cell_phi_k_phi_k)
                m_cell_phi_l_phi_l = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis_l, cell_basis_l)
                m_cell_phi_l_phi_l_inv = np.linalg.inv(m_cell_phi_l_phi_l)
                # ------------------------------------------------------------------------------------------------------
                # proj T -> F, l -> k
                # ------------------------------------------------------------------------------------------------------
                m_face_phi_l_psi_k = Integration.get_hybrid_mass_matrix_in_face(
                    cell, face, cell_basis_l, face_basis_k, passmat
                )
                m_face_psi_k_phi_l = m_face_phi_l_psi_k.T
                pi_c_f_l_k = m_face_psi_k_psi_k_inv @ m_face_psi_k_phi_l
                # ------------------------------------------------------------------------------------------------------
                # proj T -> F, k -> k
                # ------------------------------------------------------------------------------------------------------
                m_face_phi_k_psi_k = Integration.get_hybrid_mass_matrix_in_face(
                    cell, face, cell_basis_k, face_basis_k, passmat
                )
                m_face_psi_k_phi_k = m_face_phi_k_psi_k.T
                pi_c_f_k_k = m_face_psi_k_psi_k_inv @ m_face_psi_k_phi_k
                # ------------------------------------------------------------------------------------------------------
                # proj T -> F, k1 -> k
                # ------------------------------------------------------------------------------------------------------
                m_face_phi_k1_psi_k = Integration.get_hybrid_mass_matrix_in_face(
                    cell, face, cell_basis_k1, face_basis_k, passmat
                )
                m_face_psi_k_phi_k1 = m_face_phi_k1_psi_k.T
                pi_c_f_k1_k = m_face_psi_k_psi_k_inv @ m_face_psi_k_phi_k1
                # ------------------------------------------------------------------------------------------------------
                # proj T -> T, k1 -> l
                # ------------------------------------------------------------------------------------------------------
                m_cell_phi_l_phi_k1 = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis_l, cell_basis_k1)
                pi_c_c_k1_l = m_cell_phi_l_phi_l_inv @ m_cell_phi_l_phi_k1
                # ------------------------------------------------------------------------------------------------------
                m_stab_face_jump = np.zeros((face_basis_k.basis_dimension, local_problem_size))
                m_stab_face_jump[:, c0_c:c1_c] -= pi_c_f_l_k
                m_stab_face_jump[:, c0_f:c1_f] += m_face_id
                # ------------------------------------------------------------------------------------------------------
                c0_r = i * cell_basis_k1.basis_dimension
                c1_r = (i + 1) * cell_basis_k1.basis_dimension
                m_stab_face_jump -= pi_c_f_k1_k @ local_reconstruction_operator[c0_r:c1_r, :]
                m_stab_face_jump += pi_c_f_l_k @ (pi_c_c_k1_l @ local_reconstruction_operator[c0_r:c1_r, :])
                h_f = 1.0 / face.diameter
                local_stabilization_operator += h_f * m_stab_face_jump.T @ m_face_psi_k_psi_k_inv @ m_stab_face_jump
        return local_stabilization_operator
