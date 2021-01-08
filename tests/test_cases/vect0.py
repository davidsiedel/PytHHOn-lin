import os
from typing import List
from typing import Callable
import numpy as np
from numpy import ndarray as Mat
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib import rc

from tests import context
from tests.context import source

# from pythhon3d import build, solve

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 12})
rc("text", usetex=True)

import numpy as np
from typing import List
from typing import Callable
from numpy import ndarray as Mat

from parsers.geof_parser import parse_geof_file as parse_mesh
from parsers.element_types import C_cf_ref
from bases.basis import Basis
from bases.monomial import ScaledMonomial
from core.face import Face
from core.cell import Cell
from core.unknown import Unknown
from core.operators.operator import Operator
from core.operators.hdg import HDG
from core.operators.hdgs import HDGs
from core.operators.hho import HHO
from core.pressure import Pressure
from core.displacement import Displacement
from core.load import Load
from core.condensation import Condensation
from core.integration import Integration


def build(
    mesh_file: str, field_dimension: int, face_polynomial_order: int, cell_polynomial_order: int, operator_type: str,
):
    """
    ====================================================================================================================
    Description :
    ====================================================================================================================
    
    ====================================================================================================================
    Parameters :
    ====================================================================================================================
    
    ====================================================================================================================
    Exemple :
    ====================================================================================================================
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Checking polynomial order consistency
    # ------------------------------------------------------------------------------------------------------------------
    # is_polynomial_order_consistent(face_polynomial_order, cell_polynomial_order)
    # ------------------------------------------------------------------------------------------------------------------
    # Reading the mesh file, and extracting conectivity matrices
    # ------------------------------------------------------------------------------------------------------------------
    (
        problem_dimension,
        vertices,
        vertices_weights,
        cells_vertices_connectivity_matrix,
        faces_vertices_connectivity_matrix,
        cells_faces_connectivity_matrix,
        cells_connectivity_matrix,
        nsets_faces,
    ) = parse_mesh(mesh_file)
    # ------------------------------------------------------------------------------------------------------------------
    # Writing the vertices matrix as a numpy object
    # ------------------------------------------------------------------------------------------------------------------
    vertices = np.array(vertices)
    # print("vertices : \n{}".format(vertices[0:4]))
    # ------------------------------------------------------------------------------------------------------------------
    # Creating unknown object
    # ------------------------------------------------------------------------------------------------------------------
    unknown = Unknown(problem_dimension, field_dimension, cell_polynomial_order, face_polynomial_order)
    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing polynomial bases
    # ------------------------------------------------------------------------------------------------------------------
    face_basis_k = ScaledMonomial(face_polynomial_order, problem_dimension - 1)
    cell_basis_k = ScaledMonomial(face_polynomial_order, problem_dimension)
    cell_basis_l = ScaledMonomial(cell_polynomial_order, problem_dimension)
    cell_basis_k1 = ScaledMonomial(face_polynomial_order + 1, problem_dimension)
    integration_order = unknown.integration_order
    integration_order = 2 * (face_polynomial_order + 1)
    # print(integration_order)
    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing Face objects
    # ------------------------------------------------------------------------------------------------------------------
    faces = []
    for i, face_vertices_connectivity_matrix in enumerate(faces_vertices_connectivity_matrix):
        face_vertices = vertices[face_vertices_connectivity_matrix]
        face = Face(face_vertices, integration_order)
        faces.append(face)
    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing Cell objects
    # ------------------------------------------------------------------------------------------------------------------
    cells = []
    for cell_vertices_connectivity_matrix, cell_connectivity_matrix in zip(
        cells_vertices_connectivity_matrix, cells_connectivity_matrix
    ):
        cell_vertices = vertices[cell_vertices_connectivity_matrix]
        cell = Cell(cell_vertices, cell_connectivity_matrix, integration_order)
        cells.append(cell)
    # print("DONE")
    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing Elements objects
    # ------------------------------------------------------------------------------------------------------------------
    elements = []
    operators = []
    for i, cell in enumerate(cells):
        local_faces = [faces[j] for j in cells_faces_connectivity_matrix[i]]
        op = get_operator(
            operator_type, cell, local_faces, cell_basis_l, cell_basis_k, cell_basis_k1, face_basis_k, unknown
        )
        operators.append(op)
    return (
        vertices,
        faces,
        cells,
        operators,
        cells_faces_connectivity_matrix,
        cells_vertices_connectivity_matrix,
        faces_vertices_connectivity_matrix,
        nsets_faces,
        cell_basis_l,
        cell_basis_k,
        face_basis_k,
        unknown,
    )


def solve(
    vertices: Mat,
    faces: List[Face],
    cells: List[Cell],
    operators: List[Operator],
    cells_faces_connectivity_matrix: Mat,
    cells_vertices_connectivity_matrix: Mat,
    faces_vertices_connectivity_matrix: Mat,
    nsets_faces: dict,
    cell_basis_l: Basis,
    cell_basis_k: Basis,
    face_basis_k: Basis,
    unknown: Unknown,
    tangent_matrices: List[Mat],
    stabilization_parameter: float,
    boundary_conditions: dict,
    load: List[Callable],
):
    """
    ====================================================================================================================
    Description :
    ====================================================================================================================
    
    ====================================================================================================================
    Parameters :
    ====================================================================================================================
    
    ====================================================================================================================
    Exemple :
    ====================================================================================================================
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Global matrix
    # ------------------------------------------------------------------------------------------------------------------
    number_of_faces = len(faces_vertices_connectivity_matrix)
    number_of_cells = len(cells_vertices_connectivity_matrix)
    total_system_size = number_of_faces * face_basis_k.basis_dimension * unknown.field_dimension
    global_matrix = np.zeros((total_system_size, total_system_size))
    global_vector = np.zeros((total_system_size,))
    stored_matrices = []
    # ------------------------------------------------------------------------------------------------------------------
    # Global matrix
    # ------------------------------------------------------------------------------------------------------------------
    cells_indices = range(len(cells))
    for cell_index in cells_indices:
        local_cell = cells[cell_index]
        local_faces = [faces[i] for i in cells_faces_connectivity_matrix[cell_index]]
        # --------------------------------------------------------------------------------------------------------------
        # External forces
        # --------------------------------------------------------------------------------------------------------------
        number_of_local_faces = len(local_faces)
        a = unknown.field_dimension * (
            cell_basis_l.basis_dimension + number_of_local_faces * face_basis_k.basis_dimension
        )
        local_external_forces = np.zeros((a,))
        load_vector = Load(local_cell, cell_basis_l, unknown, load).load_vector
        l0 = 0
        l1 = unknown.field_dimension * cell_basis_l.basis_dimension
        local_external_forces[l0:l1] += load_vector
        connectivity = cells_faces_connectivity_matrix[cell_index]
        local_faces_indices = cells_faces_connectivity_matrix[cell_index]
        for local_face_index, global_face_index in enumerate(local_faces_indices):
            face = faces[global_face_index]
            face_reference_frame_transformation_matrix = Operator.get_face_passmat(local_cell, face)
            # for boundary_name, nset in zip(nsets, nsets.values()):
            for boundary_name, nset in zip(nsets_faces, nsets_faces.values()):
                if connectivity[local_face_index] in nset:
                    pressure = boundary_conditions[boundary_name][1]
                    pressure_vector = Pressure(
                        face, face_basis_k, face_reference_frame_transformation_matrix, unknown, pressure
                    ).pressure_vector
                else:
                    pressure_vector = Pressure(
                        face, face_basis_k, face_reference_frame_transformation_matrix, unknown
                    ).pressure_vector
            l0 = local_face_index * unknown.field_dimension * face_basis_k.basis_dimension
            l1 = (local_face_index + 1) * unknown.field_dimension * face_basis_k.basis_dimension
            local_external_forces[l0:l1] += pressure_vector
        # --------------------------------------------------------------------------------------------------------------
        # Stffness matrix
        # --------------------------------------------------------------------------------------------------------------
        tangent_matrix = tangent_matrices[cell_index]
        tangent_matrix_size = tangent_matrix.shape[0]
        total_size = tangent_matrix_size * cell_basis_k.basis_dimension
        local_mass_matrix = np.zeros((total_size, total_size))
        m_phi_phi_cell = Integration.get_cell_mass_matrix_in_cell(local_cell, cell_basis_k, cell_basis_k)
        for i in range(tangent_matrix.shape[0]):
            for j in range(tangent_matrix.shape[1]):
                m = tangent_matrix[i, j] * m_phi_phi_cell
                # ------------------------------------------------------------------------------------------------------
                l0 = i * cell_basis_k.basis_dimension
                l1 = (i + 1) * cell_basis_k.basis_dimension
                c0 = j * cell_basis_k.basis_dimension
                c1 = (j + 1) * cell_basis_k.basis_dimension
                # ------------------------------------------------------------------------------------------------------
                local_mass_matrix[l0:l1, c0:c1] += m
        local_gradient_operator = operators[cell_index].local_gradient_operator
        # print(local_gradient_operator.shape)
        local_stiffness_form = local_gradient_operator.T @ local_mass_matrix @ local_gradient_operator
        # --------------------------------------------------------------------------------------------------------------
        # Stabilization matrix
        # --------------------------------------------------------------------------------------------------------------
        local_stabilization_operator = stabilization_parameter * operators[cell_index].local_stabilization_operator
        # --------------------------------------------------------------------------------------------------------------
        # Local matrix
        # --------------------------------------------------------------------------------------------------------------
        local_matrix = local_stiffness_form + local_stabilization_operator
        # --------------------------------------------------------------------------------------------------------------
        # Static condensation
        # --------------------------------------------------------------------------------------------------------------
        (
            m_cell_cell_inv,
            m_cell_faces,
            m_faces_cell,
            m_faces_faces,
            v_cell,
            v_faces,
        ) = Condensation.get_system_decomposition(local_matrix, local_external_forces, unknown, cell_basis_l)
        # --------------------------------------------------------------------------------------------------------------
        # Static condensation
        # --------------------------------------------------------------------------------------------------------------
        m_cond, v_cond = Condensation.get_condensated_system(
            m_cell_cell_inv, m_cell_faces, m_faces_cell, m_faces_faces, v_cell, v_faces,
        )
        v_cell, m_cell_faces, m_cell_cell_inv
        stored_matrices.append((v_cell, m_cell_faces, m_cell_cell_inv))
        # --------------------------------------------------------------------------------------------------------------
        # Assembly
        # --------------------------------------------------------------------------------------------------------------
        cell_faces_connectivity_matrix = cells_faces_connectivity_matrix[cell_index]
        for local_index_col, global_index_col in enumerate(cell_faces_connectivity_matrix):
            g0c = global_index_col * face_basis_k.basis_dimension * unknown.field_dimension
            g1c = (global_index_col + 1) * face_basis_k.basis_dimension * unknown.field_dimension
            l0c = local_index_col * face_basis_k.basis_dimension * unknown.field_dimension
            l1c = (local_index_col + 1) * face_basis_k.basis_dimension * unknown.field_dimension
            global_vector[g0c:g1c] += v_cond[l0c:l1c]
            for local_index_row, global_index_row in enumerate(cell_faces_connectivity_matrix):
                g0r = global_index_row * face_basis_k.basis_dimension * unknown.field_dimension
                g1r = (global_index_row + 1) * face_basis_k.basis_dimension * unknown.field_dimension
                l0r = local_index_row * face_basis_k.basis_dimension * unknown.field_dimension
                l1r = (local_index_row + 1) * face_basis_k.basis_dimension * unknown.field_dimension
                global_matrix[g0r:g1r, g0c:g1c] += m_cond[l0r:l1r, l0c:l1c]
    # ------------------------------------------------------------------------------------------------------------------
    # Displacement enforcement through Lagrange multiplier
    # ------------------------------------------------------------------------------------------------------------------
    number_of_constrained_faces = 0
    # for boundary_name, nset in zip(nsets, nsets.values()):
    for boundary_name, nset in zip(nsets_faces, nsets_faces.values()):
        # print(boundary_name, nset)
        displacement = boundary_conditions[boundary_name][0]
        for displacement_component in displacement:
            if not displacement_component is None:
                number_of_constrained_faces += len(nset)
    lagrange_multiplyer_matrix = np.zeros(
        (
            number_of_constrained_faces * face_basis_k.basis_dimension,
            # number_of_faces * unknown.field_dimension * face_basis_k.basis_dimension,
            total_system_size,
        )
    )
    h_vector = np.zeros((number_of_constrained_faces * face_basis_k.basis_dimension,))
    iter_constrained_face = 0
    # for boundary_name, nset in zip(nsets, nsets.values()):
    for boundary_name, nset in zip(nsets_faces, nsets_faces.values()):
        for face_global_index in nset:
            face = faces[face_global_index]
            face_reference_frame_transformation_matrix = face.reference_frame_transformation_matrix
            # ----------------------------------------------------------------------------------------------------------
            m_psi_psi_face = Integration.get_face_mass_matrix_in_face(
                face, face_basis_k, face_basis_k, face_reference_frame_transformation_matrix
            )
            m_psi_psi_face_inv = np.linalg.inv(m_psi_psi_face)
            # ----------------------------------------------------------------------------------------------------------
            displacement = boundary_conditions[boundary_name][0]
            for direction, displacement_component in enumerate(displacement):
                if not displacement_component is None:
                    displacement_vector = Displacement(
                        face, face_basis_k, face_reference_frame_transformation_matrix, displacement_component
                    ).displacement_vector
                    # if False:
                    #     print("hello")
                    #     displacement_vector = np.zeros((face_basis.basis_dimension,))
                    #     if displacement_component(0) != 0:
                    #         displacement_vector[0] = 1.0
                    #     print(displacement_vector)
                    # --------------------------------------------------------------------------------------------------
                    displacement_vector = m_psi_psi_face_inv @ displacement_vector
                    # --------------------------------------------------------------------------------------------------
                    l0 = iter_constrained_face * face_basis_k.basis_dimension
                    l1 = (iter_constrained_face + 1) * face_basis_k.basis_dimension
                    c0 = (
                        face_global_index * unknown.field_dimension * face_basis_k.basis_dimension
                        + direction * face_basis_k.basis_dimension
                    )
                    c1 = (face_global_index * unknown.field_dimension * face_basis_k.basis_dimension) + (
                        direction + 1
                    ) * face_basis_k.basis_dimension
                    # --------------------------------------------------------------------------------------------------
                    lagrange_multiplyer_matrix[l0:l1, c0:c1] += np.eye(face_basis_k.basis_dimension)
                    # --------------------------------------------------------------------------------------------------
                    h_vector[l0:l1] += displacement_vector
                    iter_constrained_face += 1
    # print("h_vector : \n{}".format(h_vector))
    # print(h_vector)
    # print("lagrange_multiplyer_matrix : \n{}".format(lagrange_multiplyer_matrix))
    # ------------------------------------------------------------------------------------------------------------------
    double_lagrange = False
    # ------------------------------------------------------------------------------------------------------------------
    # If a single Lagrange multiplyier is used to enforce Dirichlet boundary conditions
    # ------------------------------------------------------------------------------------------------------------------
    if not double_lagrange:
        global_vector_2 = np.zeros((total_system_size + number_of_constrained_faces * face_basis_k.basis_dimension,))
        # --------------------------------------------------------------------------------------------------------------
        global_vector_2[:total_system_size] += global_vector
        global_vector_2[total_system_size:] += h_vector
        # --------------------------------------------------------------------------------------------------------------
        global_matrix_2 = np.zeros(
            (
                total_system_size + number_of_constrained_faces * face_basis_k.basis_dimension,
                total_system_size + number_of_constrained_faces * face_basis_k.basis_dimension,
            )
        )
        # --------------------------------------------------------------------------------------------------------------
        global_matrix_2[:total_system_size, :total_system_size] += global_matrix
        global_matrix_2[:total_system_size, total_system_size:] += lagrange_multiplyer_matrix.T
        global_matrix_2[total_system_size:, :total_system_size] += lagrange_multiplyer_matrix
    # ------------------------------------------------------------------------------------------------------------------
    # If double Lagrange multiplyiers are used to enforce Dirichlet boundary conditions
    # ------------------------------------------------------------------------------------------------------------------
    else:
        global_vector_2 = np.zeros(
            (total_system_size + 2 * (number_of_constrained_faces * face_basis_k.basis_dimension),)
        )
        # --------------------------------------------------------------------------------------------------------------
        l0 = 0
        l1 = total_system_size
        global_vector_2[l0:l1] += global_vector
        # --------------------------------------------------------------------------------------------------------------
        l0 = total_system_size
        l1 = total_system_size + (number_of_constrained_faces * face_basis_k.basis_dimension)
        global_vector_2[l0:l1] += h_vector
        # --------------------------------------------------------------------------------------------------------------
        l0 = total_system_size + (number_of_constrained_faces * face_basis_k.basis_dimension)
        l1 = total_system_size + 2 * (number_of_constrained_faces * face_basis_k.basis_dimension)
        global_vector_2[l0:l1] += h_vector
        # --------------------------------------------------------------------------------------------------------------
        global_matrix_2 = np.zeros(
            (
                total_system_size + 2 * (number_of_constrained_faces * face_basis_k.basis_dimension),
                total_system_size + 2 * (number_of_constrained_faces * face_basis_k.basis_dimension),
            )
        )
        # --------------------------------------------------------------------------------------------------------------
        l0 = 0
        l1 = total_system_size
        c0 = 0
        c1 = total_system_size
        global_matrix_2[l0:l1, c0:c1] += global_matrix
        # --------------------------------------------------------------------------------------------------------------
        l0 = 0
        l1 = total_system_size
        c0 = total_system_size
        c1 = total_system_size + (number_of_constrained_faces * face_basis_k.basis_dimension)
        global_matrix_2[l0:l1, c0:c1] += lagrange_multiplyer_matrix.T
        # --------------------------------------------------------------------------------------------------------------
        l0 = 0
        l1 = total_system_size
        c0 = total_system_size + (number_of_constrained_faces * face_basis_k.basis_dimension)
        c1 = total_system_size + 2 * (number_of_constrained_faces * face_basis_k.basis_dimension)
        global_matrix_2[l0:l1, c0:c1] += lagrange_multiplyer_matrix.T
        # --------------------------------------------------------------------------------------------------------------
        l0 = total_system_size
        l1 = total_system_size + (number_of_constrained_faces * face_basis_k.basis_dimension)
        c0 = 0
        c1 = total_system_size
        global_matrix_2[l0:l1, c0:c1] += lagrange_multiplyer_matrix
        # --------------------------------------------------------------------------------------------------------------
        l0 = total_system_size + (number_of_constrained_faces * face_basis_k.basis_dimension)
        l1 = total_system_size + 2 * (number_of_constrained_faces * face_basis_k.basis_dimension)
        c0 = 0
        c1 = total_system_size
        global_matrix_2[l0:l1, c0:c1] += lagrange_multiplyer_matrix
        # --------------------------------------------------------------------------------------------------------------
        id_lag = np.eye(number_of_constrained_faces * face_basis_k.basis_dimension)
        # --------------------------------------------------------------------------------------------------------------
        l0 = total_system_size
        l1 = total_system_size + (number_of_constrained_faces * face_basis_k.basis_dimension)
        c0 = total_system_size
        c1 = total_system_size + (number_of_constrained_faces * face_basis_k.basis_dimension)
        global_matrix_2[l0:l1, c0:c1] += id_lag
        # --------------------------------------------------------------------------------------------------------------
        l0 = total_system_size
        l1 = total_system_size + (number_of_constrained_faces * face_basis_k.basis_dimension)
        c0 = total_system_size + (number_of_constrained_faces * face_basis_k.basis_dimension)
        c1 = total_system_size + 2 * (number_of_constrained_faces * face_basis_k.basis_dimension)
        global_matrix_2[l0:l1, c0:c1] -= id_lag
        # --------------------------------------------------------------------------------------------------------------
        l0 = total_system_size + (number_of_constrained_faces * face_basis_k.basis_dimension)
        l1 = total_system_size + 2 * (number_of_constrained_faces * face_basis_k.basis_dimension)
        c0 = total_system_size
        c1 = total_system_size + (number_of_constrained_faces * face_basis_k.basis_dimension)
        global_matrix_2[l0:l1, c0:c1] -= id_lag
        # --------------------------------------------------------------------------------------------------------------
        l0 = total_system_size + (number_of_constrained_faces * face_basis_k.basis_dimension)
        l1 = total_system_size + 2 * (number_of_constrained_faces * face_basis_k.basis_dimension)
        c0 = total_system_size + (number_of_constrained_faces * face_basis_k.basis_dimension)
        c1 = total_system_size + 2 * (number_of_constrained_faces * face_basis_k.basis_dimension)
        global_matrix_2[l0:l1, c0:c1] += id_lag
    # print("global_matrix_2 : \n{}".format(global_matrix_2))
    # print("global_vector_2 : \n{}".format(global_vector_2))
    # ------------------------------------------------------------------------------------------------------------------
    # Solving the global system for faces unknowns
    # ------------------------------------------------------------------------------------------------------------------
    # print("global_matrix : \n{}".format(global_matrix_2))
    global_solution = np.linalg.solve(global_matrix_2, global_vector_2)
    # print("global_solution : \n{}".format(global_solution))
    # ------------------------------------------------------------------------------------------------------------------
    # Getting the number of vertices
    # ------------------------------------------------------------------------------------------------------------------
    number_of_quadrature_points = 0
    for cell in cells:
        number_of_quadrature_points_in_cell = cell.quadrature_points.shape[0]
        number_of_quadrature_points_in_cell = len(cell.quadrature_points)
        number_of_quadrature_points += number_of_quadrature_points_in_cell
    print(number_of_quadrature_points)
    number_of_vertices = vertices.shape[0]
    # ------------------------------------------------------------------------------------------------------------------
    quadrature_points = np.zeros((number_of_quadrature_points, unknown.problem_dimension))
    quadrature_weights = np.zeros((number_of_quadrature_points,))
    unknowns_at_vertices = [np.zeros((number_of_vertices,)) for i in range(unknown.field_dimension)]
    f_unknowns_at_vertices = [np.zeros((number_of_vertices,)) for i in range(unknown.field_dimension)]
    # unknowns_at_quadrature_points = [np.zeros((number_of_quadrature_points,)) for i in range(unknown.field_dimension)]
    unknowns_at_quadrature_points = np.zeros((number_of_quadrature_points, unknown.field_dimension))
    error_at_quadrature_points = [np.zeros((number_of_quadrature_points,)) for i in range(unknown.field_dimension)]
    vertices_weights = np.zeros((number_of_vertices,))
    f_vertices_weights = np.zeros((number_of_vertices,))
    # ------------------------------------------------------------------------------------------------------------------
    # Desassembly
    # ------------------------------------------------------------------------------------------------------------------
    quadrature_point_count = 0
    # ==================================================================================================================
    # ==================================================================================================================
    faces_indices = range(number_of_faces)
    for face_index in faces_indices:
        face_vertices_connectivity_matrix = faces_vertices_connectivity_matrix[face_index]
        face_vertices = vertices[face_vertices_connectivity_matrix]
        face = faces[face_index]
        l0 = face_index * unknown.field_dimension * face_basis_k.basis_dimension
        l1 = (face_index + 1) * unknown.field_dimension * face_basis_k.basis_dimension
        x_face = global_solution[l0:l1]
        for i, vertex in enumerate(face_vertices):
            vertex_in_face = Face.get_points_in_face_reference_frame(vertex, face.reference_frame_transformation_matrix)
            centroid_in_face = Face.get_points_in_face_reference_frame(
                face.centroid, face.reference_frame_transformation_matrix
            )
            v = face_basis_k.get_phi_vector(vertex_in_face, centroid_in_face, face.diameter)
            for direction in range(unknown.field_dimension):
                l0 = direction * face_basis_k.basis_dimension
                l1 = (direction + 1) * face_basis_k.basis_dimension
                vertex_value_vector = v * x_face[l0:l1]
                global_index = face_vertices_connectivity_matrix[i]
                l0 = global_index
                l1 = global_index + 1
                f_unknowns_at_vertices[direction][l0:l1] += np.sum(vertex_value_vector)
                f_vertices_weights[l0:l1] += 1.0
    # ==================================================================================================================
    # ==================================================================================================================
    x_cell_list, x_faces_list = [], []
    for cell_index in cells_indices:
        # --------------------------------------------------------------------------------------------------------------
        # Getting faces unknowns
        # --------------------------------------------------------------------------------------------------------------
        local_cell = cells[cell_index]
        cell_faces_connectivity_matrix = cells_faces_connectivity_matrix[cell_index]
        local_faces = [faces[i] for i in cell_faces_connectivity_matrix]
        faces_unknown_dimension = len(local_faces) * face_basis_k.basis_dimension * unknown.field_dimension
        x_faces = np.zeros((faces_unknown_dimension,))
        for local_index_col, global_index_col in enumerate(cell_faces_connectivity_matrix):
            g0c = global_index_col * face_basis_k.basis_dimension * unknown.field_dimension
            g1c = (global_index_col + 1) * face_basis_k.basis_dimension * unknown.field_dimension
            l0c = local_index_col * face_basis_k.basis_dimension * unknown.field_dimension
            l1c = (local_index_col + 1) * face_basis_k.basis_dimension * unknown.field_dimension
            x_faces[l0c:l1c] += global_solution[g0c:g1c]
        # --------------------------------------------------------------------------------------------------------------
        # Decondensation : getting cell unknowns
        # --------------------------------------------------------------------------------------------------------------
        (v_cell, m_cell_faces, m_cell_cell_inv) = stored_matrices[cell_index]
        x_cell = Condensation.get_cell_unknown(m_cell_cell_inv, m_cell_faces, v_cell, x_faces)
        x_cell_list.append(x_cell)
        x_faces_list.append(x_faces)
        x_element = np.zeros((len(x_cell) + len(x_faces),))
        p1 = cell_basis_l.basis_dimension * unknown.field_dimension
        x_element[:p1] += x_cell
        x_element[p1:] += x_faces
        # --------------------------------------------------------------------------------------------------------------
        # error draft
        # --------------------------------------------------------------------------------------------------------------
        # m_error = Integration.get_cell_mass_matrix_in_cell(local_cell, cell_basis_l, cell_basis_l)
        # m_error_inv = np.linalg.inv(m_error)
        error_vector = np.zeros((len(x_cell),))
        # for direction in range(unknown.field_dimension):
        #     l0 = direction * cell_basis_l.basis_dimension
        #     l1 = (direction + 1) * cell_basis_l.basis_dimension
        #     anal_sol = u[direction]
        #     anal_vector = Integration.get_cell_load_vector_in_cell(local_cell, cell_basis_l, anal_sol)
        #     error_vector[l0:l1] = np.abs(m_error_inv @ anal_vector - x_cell[l0:l1])
        # operators[cell_index]
        # --------------------------------------------------------------------------------------------------------------
        cell_vertices_connectivity_matrix = cells_vertices_connectivity_matrix[cell_index]
        local_vertices = vertices[cell_vertices_connectivity_matrix]
        # --------------------------------------------------------------------------------------------------------------
        for i, vertex in enumerate(local_vertices):
            v = cell_basis_l.get_phi_vector(vertex, local_cell.centroid, local_cell.diameter)
            for direction in range(unknown.field_dimension):
                l0 = direction * cell_basis_l.basis_dimension
                l1 = (direction + 1) * cell_basis_l.basis_dimension
                vertex_value_vector = v * x_cell[l0:l1]
                global_index = cell_vertices_connectivity_matrix[i]
                l0 = global_index
                l1 = global_index + 1
                unknowns_at_vertices[direction][l0:l1] += np.sum(vertex_value_vector)
                vertices_weights[l0:l1] += 1.0
        # --------------------------------------------------------------------------------------------------------------
        for i, quadrature_point in enumerate(local_cell.quadrature_points):
            v = cell_basis_l.get_phi_vector(quadrature_point, local_cell.centroid, local_cell.diameter)
            for direction in range(unknown.field_dimension):
                l0 = direction * cell_basis_l.basis_dimension
                l1 = (direction + 1) * cell_basis_l.basis_dimension
                vertex_value_vector = v * x_cell[l0:l1]
                vertex_error_vector = v * error_vector[l0:l1]
                q0 = quadrature_point_count
                q1 = quadrature_point_count + 1
                # unknowns_at_quadrature_points[direction][q0:q1] += np.sum(vertex_value_vector)
                unknowns_at_quadrature_points[quadrature_point_count, direction] += np.sum(vertex_value_vector)
                error_at_quadrature_points[direction][q0:q1] += np.sum(vertex_error_vector)
            quadrature_points[quadrature_point_count] += quadrature_point
            quadrature_weights[quadrature_point_count] += local_cell.quadrature_weights[i]
            quadrature_point_count += 1
    # ------------------------------------------------------------------------------------------------------------------
    # Global matrix
    # ------------------------------------------------------------------------------------------------------------------
    for direction in range(unknown.field_dimension):
        unknowns_at_vertices[direction] = unknowns_at_vertices[direction] / vertices_weights
        f_unknowns_at_vertices[direction] = f_unknowns_at_vertices[direction] / f_vertices_weights
    return (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, error_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
        # unknowns_at_faces
    )


def is_polynomial_order_consistent(face_polynomial_order: int, cell_polynomial_order: int):
    """
    ====================================================================================================================
    Description :
    ====================================================================================================================
    
    ====================================================================================================================
    Parameters :
    ====================================================================================================================
    
    ====================================================================================================================
    Exemple :
    ====================================================================================================================
    """
    if not face_polynomial_order in [cell_polynomial_order - 1, cell_polynomial_order, cell_polynomial_order + 1]:
        raise ValueError(
            "The face polynomial order must be the same order as the cell polynomial order or one order lower or greater"
        )


def get_operator(
    operator_type: str,
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
        Description :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        
        ================================================================================================================
        Exemple :
        ================================================================================================================
        """
    if operator_type == "HDG":
        op = HDG(cell, faces, cell_basis_l, cell_basis_k, face_basis_k, unknown)
    elif operator_type == "HDGs":
        op = HDGs(cell, faces, cell_basis_l, cell_basis_k, face_basis_k, unknown)
    elif operator_type == "HHO":
        op = HHO(cell, faces, cell_basis_l, cell_basis_k, cell_basis_k1, face_basis_k, unknown)
    else:
        raise NameError("The specified operator does not exist")
    return op


def solve_2D_incompressible_problem(
    number_of_elements: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    stabilization_parameter: int,
):
    field_dimension = 2
    mesh_file = os.path.join(source, "meshes/2D/c2d3_{}.geof".format(number_of_elements))
    create_plot(mesh_file, number_of_elements)
    # ------------------------------------------------------------------------------------------------------------------
    pressure_left = [None, None]
    pressure_right = [None, None]
    pressure_top = [None, None]
    pressure_bottom = [None, None]
    # ------------------------------------------------------------------------------------------------------------------
    # u_0 = lambda x: (np.sin(np.pi * x[0])) * (np.sin(np.pi * x[1])) + x[0] / (2.0 * lam)
    u_0 = lambda x: 0.0
    f_0 = lambda x: -2.0 * ((np.pi) ** 2) * (np.sin(np.pi * x[0])) * (np.sin(np.pi * x[1]))
    # f_0 = lambda x: -(2.0 / ((np.pi) ** 2)) * (np.sin(np.pi * x[0])) * (np.sin(np.pi * x[1]))
    # anal_sol_x = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + 1.0 / (2.0 * lam) * x[0]
    # anal_sol_y = lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]) + 1.0 / (2.0 * lam) * x[1]
    # ------------------------------------------------------------------------------------------------------------------
    dirichlet_x_bottom = lambda s: 0.0
    # ------------------------------------------------------------------------------------------------------------------
    dirichlet_x_top = lambda s: 0.0
    # ------------------------------------------------------------------------------------------------------------------
    dirichlet_x_left = lambda s: 0.0
    # ------------------------------------------------------------------------------------------------------------------
    dirichlet_x_right = lambda s: 0.0
    # ------------------------------------------------------------------------------------------------------------------
    # load_x = lambda x: 2.0 * (np.pi ** 2) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    # load_y = lambda x: 2.0 * (np.pi ** 2) * np.cos(np.pi * x[0]) * np.cos(np.pi * x[1])
    displacement_left = [u_0, u_0]
    displacement_right = [u_0, u_0]
    displacement_top = [u_0, u_0]
    displacement_bottom = [u_0, u_0]
    # ------------------------------------------------------------------------------------------------------------------
    load = [f_0, u_0]
    # ------------------------------------------------------------------------------------------------------------------
    boundary_conditions = {
        "RIGHT": (displacement_right, pressure_right),
        "LEFT": (displacement_left, pressure_left),
        "TOP": (displacement_top, pressure_top),
        "BOTTOM": (displacement_bottom, pressure_bottom),
    }
    # ------------------------------------------------------------------------------------------------------------------
    (
        vertices,
        faces,
        cells,
        operators,
        cells_faces_connectivity_matrix,
        cells_vertices_connectivity_matrix,
        faces_vertices_connectivity_matrix,
        nsets_faces,
        cell_basis_l,
        cell_basis_k,
        face_basis_k,
        unknown,
    ) = build(mesh_file, field_dimension, face_polynomial_order, cell_polynomial_order, operator_type)
    # ------------------------------------------------------------------------------------------------------------------
    tangent_matrix = np.array([[1.0, 0.0], [0.0, 1.0],])
    tangent_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    tangent_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    tangent_matrices = [tangent_matrix for i in range(len(cells))]
    # ------------------------------------------------------------------------------------------------------------------
    (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, error_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    ) = solve(
        vertices,
        faces,
        cells,
        operators,
        cells_faces_connectivity_matrix,
        cells_vertices_connectivity_matrix,
        faces_vertices_connectivity_matrix,
        nsets_faces,
        cell_basis_l,
        cell_basis_k,
        face_basis_k,
        unknown,
        tangent_matrices,
        stabilization_parameter,
        boundary_conditions,
        load,
    )
    # ==================================================================================================================
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    # ==================================================================================================================
    x, y = quadrature_points.T
    # x, y = vertices.T
    data = ax0.tricontourf(x, y, unknowns_at_quadrature_points[:, 0], cmap=cm.binary)
    # data = ax0.tricontourf(x, y, unknowns_at_vertices[0], cmap=cm.binary)
    # data = ax0.tricontourf(x, y, unknowns_at_vertices[0], cmap=cm.binary, levels=np.linspace(0.0, 1.0, 1000))
    # data = ax.tricontourf(x, y, unknowns_at_quadrature_points[1], cmap=cm.binary)
    # data = ax.tricontourf(x, y, unknowns_at_quadrature_points[0], vmin=vmin, vmax=vmax, cmap=cm.binary)
    ax0.set_xlim(0.0, 1.0)
    ax0.set_ylim(0.0, 1.0)
    ax0.get_xaxis().set_visible(False)
    ax0.get_yaxis().set_visible(False)
    ax0.set_xlabel("map of the domain $\Omega$")
    ax0.set_title(
        "{}, $k = {}, l = {}$, stab $= {}$".format(
            operator_type, face_polynomial_order, cell_polynomial_order, stabilization_parameter
        )
    )
    cbar = fig.colorbar(data, ax=ax0)
    cbar.set_label("{} solution".format(operator_type), rotation=270, labelpad=15.0)
    # ==================================================================================================================
    x = np.linspace(0.0, 1.0, 1000)
    y = np.linspace(0.0, 1.0, 1000)
    X, Y = np.meshgrid(x, y)
    anal_sol_x = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    # Z = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * X))
    Z = np.sin(np.pi * X) * np.sin(np.pi * Y)
    nb = 500
    x = np.linspace(0.0, 1.0, nb)
    pts = []
    for i in range(nb):
        for j in range(nb):
            pts.append(np.array([x[i], x[j]]))
    vmax = max([anal_sol_x(pt) for pt in pts])
    vmin = min([anal_sol_x(pt) for pt in pts])
    # vmax = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * 0.75))
    # vmin = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * 0.25))
    col_levels = 10
    # CS = ax1.contourf(X, Y, Z, col_levels, vmin=vmin, vmax=vmax, cmap=cm.binary)
    CS = ax1.contourf(X, Y, Z, col_levels, cmap=cm.binary)
    m = plt.cm.ScalarMappable(cmap=cm.binary)
    m.set_array(Z)
    # m.set_clim(vmin, vmax)
    # cbar = fig.colorbar(m, boundaries=np.linspace(vmin, vmax, col_levels), ax=ax1)
    cbar = fig.colorbar(m, ax=ax1)
    cbar.set_label("Analytical solution", rotation=270, labelpad=15.0)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_xlabel("map of the domain $\Omega$")
    ax1.set_title("Analytical Solution")
    # ==================================================================================================================
    nb = 500
    x = np.linspace(0.0, 1.0, nb)
    pts = []
    for i in range(nb):
        for j in range(nb):
            pts.append(np.array([x[i], x[j]]))
    vmax = max([anal_sol_x(pt) for pt in pts])
    x, y = quadrature_points.T
    error = unknowns_at_quadrature_points[:, 0] - np.array([anal_sol_x(quad_p) for quad_p in quadrature_points])
    error = np.abs(error) / vmax * 100.0
    data = ax2.tricontourf(x, y, error, cmap=cm.binary)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_xlabel("map of the domain $\Omega$")
    ax2.set_title("Error")
    cbar = fig.colorbar(data, ax=ax2)
    cbar.set_label("Relative error (percent)", rotation=270, labelpad=15.0)
    plt.show()
    return (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, error_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    )


def create_plot(file_path: str, number_of_elements):
    with open(file_path, "w") as mesh_file:
        number_of_nodes = (number_of_elements + 1) ** 2
        nb_nd = number_of_elements + 1
        mesh_file.write("***geometry\n**node\n{} 2\n".format(number_of_nodes))
        segmentation = np.linspace(0.0, 1.0, number_of_elements + 1)
        count = 1
        for point_Y in segmentation:
            for point_X in segmentation:
                mesh_file.write("{} {} {}\n".format(count, point_X, point_Y))
                count += 1
        nb_el = 2 * (number_of_elements * number_of_elements)
        mesh_file.write("**element\n{}\n".format(nb_el))
        count = 1
        for count_node in range(number_of_elements):
            for count_line in range(number_of_elements):
                count_node_eff = count_node + 1
                count_line_eff = count_line
                # p1 = (count_line * number_of_nodes) + count_node
                # p2 = (count_line * number_of_nodes) + count_node + 1
                # p3 = ((count_line + 1) * number_of_nodes) + count_node
                p1 = count_node_eff + (count_line_eff * nb_nd)
                p2 = count_node_eff + 1 + (count_line_eff * nb_nd)
                p3 = count_node_eff + nb_nd + (count_line_eff * nb_nd)
                #
                # mesh_file.write("{} c2d3 {} {} {}\n".format(count, p1 + 1, p2 + 1, p3 + 1))
                mesh_file.write("{} c2d3 {} {} {}\n".format(count, p1, p2, p3))
                count += 1
                #
                # q1 = (count_line * number_of_nodes) + count_node + 1
                # q2 = ((count_line + 1) * number_of_nodes) + count_node
                # q3 = ((count_line + 2) * number_of_nodes) + count_node
                q1 = count_node_eff + 1 + (count_line_eff * nb_nd)
                q2 = count_node_eff + nb_nd + (count_line_eff * nb_nd)
                q3 = count_node_eff + nb_nd + 1 + (count_line_eff * nb_nd)
                #
                # mesh_file.write("{} c2d3 {} {} {}\n".format(count, q1 + 1, q2 + 1, q3 + 1))
                mesh_file.write("{} c2d3 {} {} {}\n".format(count, q1, q3, q2))
                count += 1
        # mesh_file.write("***group\n**nset LEFT\n1\n**nset RIGHT\n{}\n***return".format(number_of_nodes))
        mesh_file.write("***group\n")
        mesh_file.write("**nset BOTTOM\n")
        count = 1
        for point_Y in segmentation:
            for point_X in segmentation:
                if point_Y == 0.0:
                    mesh_file.write(" {}".format(count))
                count += 1
        mesh_file.write("\n")
        mesh_file.write("**nset TOP\n")
        count = 1
        for point_Y in segmentation:
            for point_X in segmentation:
                if point_Y == 1.0:
                    mesh_file.write(" {}".format(count))
                count += 1
        mesh_file.write("\n")
        mesh_file.write("**nset LEFT\n")
        count = 1
        for point_Y in segmentation:
            for point_X in segmentation:
                if point_X == 0.0:
                    mesh_file.write(" {}".format(count))
                count += 1
        mesh_file.write("\n")
        mesh_file.write("**nset RIGHT\n")
        count = 1
        for point_Y in segmentation:
            for point_X in segmentation:
                if point_X == 1.0:
                    mesh_file.write(" {}".format(count))
                count += 1
        mesh_file.write("\n")
        mesh_file.write("***return")
    return


number_of_elements = 10
face_polynomial_order = 1
cell_polynomial_order = 1
operator_type = "HDG"
stabilization_parameter = 1.0

solve_2D_incompressible_problem(
    number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter
)

