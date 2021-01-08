import os
from typing import List
from typing import Callable
import numpy as np
from numpy import ndarray as Mat

import subprocess

# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
# import matplotlib.patches as mpatches
# import matplotlib.colors as colors
# import matplotlib.cbook as cbook
# import matplotlib.cm as cm
# import matplotlib.mlab as mlab
# from matplotlib.colors import LinearSegmentedColormap
# import matplotlib.tri as mtri
# from matplotlib import rc
# rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 12})
# rc("text", usetex=True)

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.sparse import dia_matrix
import scipy.sparse.linalg as spla

import mgis
import mgis.behaviour as mgis_bv

import sys, os
from pathlib import Path

current_folder = Path(os.path.dirname(os.path.abspath(__file__)))
source = current_folder.parent.parent
package_folder = os.path.join(source, "pythhon3d")
sys.path.insert(0, package_folder)

# from tests import context
# from tests.context import source

# from pythhon3d import build, solve


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

# lam = 121153.84615384616
# lam = 121153000000000000.84615384616
# lam = 1.84615384616
# mu = 80769.23076923077
#
# mu = 0.375
# lam = 7.5e1
# lam, mu = 1.0, 1.0
# E = 1.0
E = 1.124999981250001
# nu = 0.4999999999999999
nu = 0.49999997500000126
lam = (E * nu) / ((1.0 + nu) * (1 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
# mu = 0.375
# lam = 7.5e6

print("lam : {}".format(lam))
print("mu : {}".format(mu))

u_0 = lambda x: (np.sin(np.pi * x[0])) * (np.sin(np.pi * x[1])) + x[0] / (2.0 * lam)
u_1 = lambda x: (np.cos(np.pi * x[0])) * (np.cos(np.pi * x[1])) + x[1] / (2.0 * lam)

u = [u_0, u_1]

f_0 = lambda x: (2.0 * (np.pi) ** 2) * (np.sin(np.pi * x[0])) * (np.sin(np.pi * x[1]))
f_1 = lambda x: (2.0 * (np.pi) ** 2) * (np.cos(np.pi * x[0])) * (np.cos(np.pi * x[1]))

f = [f_0, f_1]


def build(
    mesh_file: str,
    field_dimension: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
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
    print("faces done")
    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing Cell objects
    # ------------------------------------------------------------------------------------------------------------------
    cells = []
    quad_points = []
    quad_weights = []
    num_quad_in_mesh = 0
    for cell_vertices_connectivity_matrix, cell_connectivity_matrix in zip(
        cells_vertices_connectivity_matrix, cells_connectivity_matrix
    ):
        cell_vertices = vertices[cell_vertices_connectivity_matrix]
        cell = Cell(cell_vertices, cell_connectivity_matrix, integration_order)
        cells.append(cell)
        quad_points.append(cell.quadrature_points)
        quad_weights.append(cell.quadrature_weights)
        num_quad_in_mesh += len(cell.quadrature_points)
    mesh_quadrature_points = np.zeros((num_quad_in_mesh, problem_dimension))
    mesh_quadrature_weights = np.zeros((num_quad_in_mesh,))
    for iteration, iter_cell in enumerate(cells):
        r0 = iteration * len(iter_cell.quadrature_points)
        r1 = (iteration + 1) * len(iter_cell.quadrature_points)
        mesh_quadrature_points[r0:r1, :] += iter_cell.quadrature_points
        # print(iter_cell.quadrature_weights[:, 0])
        mesh_quadrature_weights[r0:r1] += iter_cell.quadrature_weights[:, 0]
    print("cells done")
    # print("DONE")
    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing Elements objects
    # ------------------------------------------------------------------------------------------------------------------
    operators = []
    for i, cell in enumerate(cells):
        local_faces = [faces[j] for j in cells_faces_connectivity_matrix[i]]
        op = get_operator(
            operator_type, cell, local_faces, cell_basis_l, cell_basis_k, cell_basis_k1, face_basis_k, unknown
        )
        operators.append(op)
    print("ops done")
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
        mesh_quadrature_points,
        mesh_quadrature_weights,
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
    mesh_quadrature_points: Mat,
    mesh_quadrature_weights: Mat,
    material_data_manager,
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
    print("transfer done")
    total_system_size = number_of_faces * face_basis_k.basis_dimension * unknown.field_dimension
    global_matrix = np.zeros((total_system_size, total_system_size))
    global_vector = np.zeros((total_system_size,))
    print("global matrices done")
    stored_matrices = []
    # ------------------------------------------------------------------------------------------------------------------
    # internal_forces = np.zeros((number_of_cells * unknown.field_dimension,))
    # external_forces = np.zeros((number_of_cells* unknown.field_dimension,))
    internal_forces = np.zeros((total_system_size,))
    external_forces = np.zeros((total_system_size,))
    # ------------------------------------------------------------------------------------------------------------------
    # Global matrix
    # ------------------------------------------------------------------------------------------------------------------
    cells_indices = range(len(cells))
    quad_point_count = 0
    for cell_index in cells_indices:
        local_cell = cells[cell_index]
        local_faces = [faces[i] for i in cells_faces_connectivity_matrix[cell_index]]
        # --------------------------------------------------------------------------------------------------------------
        # External forces
        # --------------------------------------------------------------------------------------------------------------
        number_of_local_faces = len(local_faces)
        a = (
            cell_basis_l.basis_dimension * unknown.field_dimension
            + number_of_local_faces * face_basis_k.basis_dimension * unknown.field_dimension
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
            for boundary_name, nset in zip(nsets_faces, nsets_faces.values()):
                if global_face_index in nset:
                    pressure = boundary_conditions[boundary_name][1]
                    pressure_vector = Pressure(
                        face, face_basis_k, face_reference_frame_transformation_matrix, unknown, pressure
                    ).pressure_vector
                    l0 = (
                        unknown.field_dimension * cell_basis_l.basis_dimension
                        + local_face_index * unknown.field_dimension * face_basis_k.basis_dimension
                    )
                    l1 = (
                        unknown.field_dimension * cell_basis_l.basis_dimension
                        + (local_face_index + 1) * unknown.field_dimension * face_basis_k.basis_dimension
                    )
                    local_external_forces[l0:l1] += pressure_vector
        # --------------------------------------------------------------------------------------------------------------
        # Stffness matrix : internal forces
        # --------------------------------------------------------------------------------------------------------------
        elem_matrix_size = (
            cell_basis_l.basis_dimension * unknown.field_dimension
            + (face_basis_k.basis_dimension * unknown.field_dimension) * number_of_local_faces
        )
        elem_matrix = np.zeros((elem_matrix_size, elem_matrix_size))
        tangent_matrix_size = 3
        local_mass_matrix_size = tangent_matrix_size * cell_basis_k.basis_dimension
        local_mass_matrix = np.zeros((local_mass_matrix_size, local_mass_matrix_size))
        for local_quad_point, local_quad_weight in zip(local_cell.quadrature_points, local_cell.quadrature_weights):
            x_Q_c = local_quad_point
            x_c = local_cell.centroid
            v_c = local_cell.diameter
            w_Q_c = local_quad_weight
            tangent_matrix_mgis = material_data_manager.K[quad_point_count]
            # print("tangent_matrix_mgis : \n{}".format(tangent_matrix_mgis))
            tangent_matrix = tangent_matrix_mgis[[[0], [1], [3]], [0, 1, 3]]
            # ----------------------------------------------------------------------------------------------------------
            phi_vector_0 = cell_basis_k.get_phi_vector(x_Q_c, x_c, v_c)
            number_of_components = phi_vector_0.shape[0]
            phi_vector_0 = np.resize(phi_vector_0, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            phi_vector_1 = cell_basis_k.get_phi_vector(x_Q_c, x_c, v_c)
            number_of_components = phi_vector_1.shape[0]
            phi_vector_1 = np.resize(phi_vector_1, (1, number_of_components))
            # ----------------------------------------------------------------------------------------------------------
            m_phi_phi_cell = w_Q_c * phi_vector_0.T @ phi_vector_1
            for i in range(tangent_matrix.shape[0]):
                for j in range(tangent_matrix.shape[1]):
                    if i == 2 and j == 2:
                        coef = 2.0
                    else:
                        coef = 1.0
                    m = coef * tangent_matrix[i, j] * m_phi_phi_cell
                    # --------------------------------------------------------------------------------------------------
                    l0 = i * cell_basis_k.basis_dimension
                    l1 = (i + 1) * cell_basis_k.basis_dimension
                    c0 = j * cell_basis_k.basis_dimension
                    c1 = (j + 1) * cell_basis_k.basis_dimension
                    # --------------------------------------------------------------------------------------------------
                    local_mass_matrix[l0:l1, c0:c1] += m
            local_gradient_operator_init = operators[cell_index].local_gradient_operator
            r0 = 0 * (cell_basis_k.basis_dimension)
            r1 = 3 * (cell_basis_k.basis_dimension)
            local_gradient_operator = local_gradient_operator_init[r0:r1, :]
            local_stiffness_form = local_gradient_operator.T @ local_mass_matrix @ local_gradient_operator
            # --------------------------------------------------------------------------------------------------
            # Stabilization matrix
            # --------------------------------------------------------------------------------------------------
            local_stabilization_operator = stabilization_parameter * operators[cell_index].local_stabilization_operator
            # --------------------------------------------------------------------------------------------------
            # Local matrix
            # --------------------------------------------------------------------------------------------------
            elem_matrix += local_stiffness_form + local_stabilization_operator
            quad_point_count += 1
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
        ) = Condensation.get_system_decomposition(elem_matrix, local_external_forces, unknown, cell_basis_l)
        # --------------------------------------------------------------------------------------------------------------
        # Static condensation
        # --------------------------------------------------------------------------------------------------------------
        m_cond, v_cond = Condensation.get_condensated_system(
            m_cell_cell_inv,
            m_cell_faces,
            m_faces_cell,
            m_faces_faces,
            v_cell,
            v_faces,
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
    print("ops on cells done")
    # ------------------------------------------------------------------------------------------------------------------
    # Displacement enforcement through Lagrange multiplier
    # ------------------------------------------------------------------------------------------------------------------
    number_of_constrained_faces = 0
    for boundary_name, nset in zip(nsets_faces, nsets_faces.values()):
        displacement = boundary_conditions[boundary_name][0]
        for displacement_component in displacement:
            if not displacement_component is None:
                number_of_constrained_faces += len(nset)
    lagrange_multiplyer_matrix = np.zeros(
        (
            number_of_constrained_faces * face_basis_k.basis_dimension,
            total_system_size,
        )
    )
    h_vector = np.zeros((number_of_constrained_faces * face_basis_k.basis_dimension,))
    iter_constrained_face = 0
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
        print("{} treated".format(boundary_name))
    # ------------------------------------------------------------------------------------------------------------------
    double_lagrange = False
    # ------------------------------------------------------------------------------------------------------------------
    # If a single Lagrange multiplyier is used to enforce Dirichlet boundary conditions
    # ------------------------------------------------------------------------------------------------------------------
    if not double_lagrange:
        print("initializing big matrix")
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
        print("filling system matrix")
        # --------------------------------------------------------------------------------------------------------------
        global_matrix_2[:total_system_size, :total_system_size] = global_matrix
        print("filled system matrix")
        global_matrix_2[:total_system_size, total_system_size:] = lagrange_multiplyer_matrix.T
        print("filled Lagrange matrix")
        global_matrix_2[total_system_size:, :total_system_size] = lagrange_multiplyer_matrix
        print("completing big matrix")
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
    print("ops on faces done")
    print("COND : {}".format(np.linalg.cond(global_matrix_2)))
    # ------------------------------------------------------------------------------------------------------------------
    # Solving the global system for faces unknowns
    # ------------------------------------------------------------------------------------------------------------------
    # global_solution = np.linalg.solve(global_matrix_2, global_vector_2)
    # print(global_matrix_2.shape)
    # M2 = spla.spilu(global_matrix_2)
    # M_x = lambda x: M2.solve(x)
    # M = spla.LinearOperator(global_matrix_2.shape, M_x)
    # global_solution = spla.gmres(global_matrix_2, global_vector_2, M=M)
    global_matrix_2_sparse = csr_matrix(global_matrix_2)
    # global_matrix_2_sparse = dia_matrix(global_matrix_2)
    global_solution = spsolve(global_matrix_2_sparse, global_vector_2)
    print("system solved")
    # ------------------------------------------------------------------------------------------------------------------
    global_unkown = global_solution[:total_system_size]
    internal_forces = global_matrix @ (global_solution[:total_system_size]).T
    # external_forces = global_vector
    # print("res : {}".format(internal_forces - global_vector))
    q_p_count = 0
    strains = []
    for cell_index in range(number_of_cells):
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
        (v_cell, m_cell_faces, m_cell_cell_inv) = stored_matrices[cell_index]
        x_cell = Condensation.get_cell_unknown(m_cell_cell_inv, m_cell_faces, v_cell, x_faces)
        x_element = np.zeros((len(x_cell) + len(x_faces),))
        p1 = cell_basis_l.basis_dimension * unknown.field_dimension
        x_element[:p1] += x_cell
        x_element[p1:] += x_faces
        grad_op = operators[cell_index].local_gradient_operator
        for q_p in local_cell.quadrature_points:
            v_k = cell_basis_k.get_phi_vector(q_p, local_cell.centroid, local_cell.diameter)
            strain = np.zeros((4,))
            l0_k = 0 * cell_basis_k.basis_dimension
            l1_k = 1 * cell_basis_k.basis_dimension
            strain[0] += v_k.T @ (grad_op[l0_k:l1_k] @ x_element)
            l0_k = 1 * cell_basis_k.basis_dimension
            l1_k = 2 * cell_basis_k.basis_dimension
            strain[1] += v_k.T @ (grad_op[l0_k:l1_k] @ x_element)
            strain[2] += 0.0
            l0_k = 2 * cell_basis_k.basis_dimension
            l1_k = 3 * cell_basis_k.basis_dimension
            strain[3] += v_k.T @ (grad_op[l0_k:l1_k] @ x_element)
            strains.append(strain)
            q_p_count += 1
    return global_unkown, internal_forces, global_vector, strains


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
    stabilization_parameter: float,
):
    field_dimension = 2
    mesh_file = os.path.join(source, "meshes/2D/c2d3_{}_cooke.geof".format(number_of_elements))
    triangles = create_plot_cooke(mesh_file, number_of_elements)
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
        mesh_quadrature_points,
        mesh_quadrature_weights,
    ) = build(mesh_file, field_dimension, face_polynomial_order, cell_polynomial_order, operator_type)
    # ------------------------------------------------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------------------------------------------------
    number_of_faces = len(faces)
    total_system_size = number_of_faces * face_basis_k.basis_dimension * unknown.field_dimension
    # ------------------------------------------------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------------------------------------------------
    lib = "/Users/davidsiedel/Projects/PytHHOn3D/bhv/src/libBehaviour.dylib"
    h = mgis_bv.Hypothesis.PlaneStrain
    b = mgis_bv.load(lib, "Elasticity", h)
    m = mgis_bv.MaterialDataManager(b, len(mesh_quadrature_points))
    it = mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator
    m.s0.setMaterialProperty("YoungModulus", 1.12499998125000100)
    m.s0.setMaterialProperty("PoissonRatio", 0.49999997500000126)
    m.s1.setMaterialProperty("YoungModulus", 1.12499998125000100)
    m.s1.setMaterialProperty("PoissonRatio", 0.49999997500000126)
    T = 293.15 * np.ones(len(mesh_quadrature_points))
    Ts = mgis_bv.MaterialStateManagerStorageMode.ExternalStorage
    mgis_bv.setExternalStateVariable(m.s0, "Temperature", T, Ts)
    mgis_bv.setExternalStateVariable(m.s1, "Temperature", T, Ts)
    it = mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator
    for quad_point_iteration in range(len(mesh_quadrature_points)):
        m.s0.gradients[quad_point_iteration] = np.zeros((4,))
        m.s0.thermodynamic_forces[quad_point_iteration] = np.zeros((4,))
        m.s1.gradients[quad_point_iteration] = np.zeros((4,))
    # dt = 0.0
    # mgis_bv.integrate(m, it, dt, 0, m.n)
    # mgis_bv.update(m)
    # print(m.K[0])
    # ------------------------------------------------------------------------------------------------------------------
    # TABLE
    # ------------------------------------------------------------------------------------------------------------------
    pressure_list = [[None, lambda x: 1.0 * x / 16.0] for x in [1.0, 1.2, 1.3]]
    load_list = [[lambda x: 0.0, lambda x: 0.0] for x in [1.0, 1.2, 1.3]]
    eps = 0.0
    Delta_unknown_at_step = np.zeros((total_system_size,))
    # ------------------------------------------------------------------------------------------------------------------
    for pressure_step, load_step in zip(pressure_list, load_list):
        # --------------------------------------------------------------------------------------------------------------
        fixed = [lambda x: 0.0, lambda x: 0.0]
        null = [None, None]
        # --------------------------------------------------------------------------------------------------------------
        load = load_step
        # --------------------------------------------------------------------------------------------------------------
        boundary_conditions = {
            "RIGHT": (null, pressure_step),
            "LEFT": (fixed, null),
            "TOP": (null, null),
            "BOTTOM": (null, null),
            "APPLI": (null, null),
        }
        # --------------------------------------------------------------------------------------------------------------
        # strain_increment = np.zeros((4,))
        # stress_increment = np.zeros((4,))
        # delta_unknown_at_iteration = Delta_unknown_at_step
        delta_unknown_at_iteration = np.zeros((total_system_size,))
        for mech_iteration in range(1):
            m.s0.setMaterialProperty("YoungModulus", 1.12499998125000100)
            m.s0.setMaterialProperty("PoissonRatio", 0.49999997500000126)
            m.s1.setMaterialProperty("YoungModulus", 1.12499998125000100)
            m.s1.setMaterialProperty("PoissonRatio", 0.49999997500000126)
            T = 293.15 * np.ones(len(mesh_quadrature_points))
            Ts = mgis_bv.MaterialStateManagerStorageMode.ExternalStorage
            mgis_bv.setExternalStateVariable(m.s0, "Temperature", T, Ts)
            mgis_bv.setExternalStateVariable(m.s1, "Temperature", T, Ts)
            mgis_bv.integrate(m, it, 0.0, 0, m.n)  # args are(mat_data, integ_type, dt=0.0, start = 0, end = m.n)
            print("stress after integ : {}".format(m.s1.thermodynamic_forces[0]))
            # for quad_point_iteration in range(len(mesh_quadrature_points)):
            #     m.s1.gradients[quad_point_iteration] += strain_increment
            # mgis_bv.integrate(m, it, 0.0, 0, m.n)  # args are(mat_data, integ_type, dt=0.0, start = 0, end = m.n)
            # mgis_bv.update(m)
            # ----------------------------------------------------------------------------------------------------------
            (global_unkown, internal_forces, external_forces, strains) = solve(
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
                mesh_quadrature_points,
                mesh_quadrature_weights,
                m,
                stabilization_parameter,
                boundary_conditions,
                load,
            )
            eps_vector = eps * np.ones(internal_forces.shape)
            residual = internal_forces - external_forces
            # print("res : {}".format(residual))
            if (np.abs(residual) < eps_vector).all():
                break
            else:
                for q_p_count in range(len(mesh_quadrature_points)):
                    m.s1.gradients[q_p_count] += strains[q_p_count]
                # q_p_count = 0
                # delta_unknown_at_iteration += global_unkown
                # for cell_index in range(len(cells)):
                #     cell = cells[cell_index]
                #     grad_op = operators[cell_index].local_gradient_operator
                #     for q_p in cell.quadrature_points:
                #         v_k = cell_basis_k.get_phi_vector(q_p, cell.centroid, cell.diameter)
                #         strain = np.zeros((4,))
                #         strain[0] += v_k.T @ grad_op[l0_k:l1_k]
    # ==================================================================================================================
    # ==================================================================================================================
    # PLOTTTTTTtTTTTTTTTt
    # ==================================================================================================================
    # ==================================================================================================================
    fig, (ax0, ax1, ax0d) = plt.subplots(nrows=1, ncols=3, sharex=True)
    # ==================================================================================================================
    # X, Y = vertices.T + f_unknowns_at_vertices.T
    X, Y = vertices.T
    # x, y = quadrature_points.T + unknowns_at_quadrature_points.T
    x, y = quadrature_points.T
    triang = mtri.Triangulation(X, Y, triangles)
    # datad = ax0d.scatter(x, y, c=lam * div_at_quadrature_points, cmap=cm.binary, s=20)
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    perso = LinearSegmentedColormap.from_list("perso", colors, N=1000)
    vmin = min(div_at_quadrature_points[:])
    vmax = max(div_at_quadrature_points[:])
    levels = np.linspace(vmin, vmax, 1000, endpoint=True)
    ticks = np.linspace(vmin, vmax, 10, endpoint=True)
    datad = ax0d.tricontourf(x, y, div_at_quadrature_points[:], cmap=perso, levels=levels)
    ax0d.triplot(triang, linewidth=0.2, color="grey")
    ax0d.get_xaxis().set_visible(False)
    ax0d.get_yaxis().set_visible(False)
    ax0d.set_xlabel("map of the domain $\Omega$")
    ax0d.set_title("{}, $k = {}, l = {}$".format(operator_type, face_polynomial_order, cell_polynomial_order))
    cbar = fig.colorbar(datad, ax=ax0d, ticks=ticks)
    cbar.set_label("divergence of the displacement".format(operator_type), rotation=270, labelpad=15.0)
    # ------------------------------------------------------------------------------------------------------------------
    data0 = ax0.tricontourf(x, y, unknowns_at_quadrature_points[:, 0], cmap=cm.binary)
    # data0 = ax0.tricontourf(X, Y, unknowns_at_vertices[:, 0], cmap=cm.binary)
    ax0.triplot(triang, linewidth=0.2, color="grey")
    ax0.get_xaxis().set_visible(False)
    ax0.get_yaxis().set_visible(False)
    ax0.set_xlabel("map of the domain $\Omega$")
    ax0.set_title("{}, $k = {}, l = {}$".format(operator_type, face_polynomial_order, cell_polynomial_order))
    cbar = fig.colorbar(data0, ax=ax0)
    cbar.set_label("{} solution, $x$-displacement".format(operator_type), rotation=270, labelpad=15.0)
    # ------------------------------------------------------------------------------------------------------------------
    data1 = ax1.tricontourf(x, y, unknowns_at_quadrature_points[:, 1], cmap=cm.binary)
    # data1 = ax1.tricontourf(X, Y, unknowns_at_vertices[:, 1], cmap=cm.binary)
    ax1.triplot(triang, linewidth=0.2, color="grey")
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_xlabel("map of the domain $\Omega$")
    ax1.set_title("{}, $k = {}, l = {}$".format(operator_type, face_polynomial_order, cell_polynomial_order))
    cbar = fig.colorbar(data1, ax=ax1)
    cbar.set_label("{} solution, $y$-displacement".format(operator_type), rotation=270, labelpad=15.0)
    # ==================================================================================================================
    plt.show()
    return (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, error_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    )


def compute_strain_vector():
    return


def create_plot_cooke(file_path: str, number_of_elements):
    with open(file_path, "w") as mesh_file:
        number_of_nodes = (number_of_elements + 1) ** 2
        nb_nd = number_of_elements + 1
        mesh_file.write("***geometry\n**node\n{} 2\n".format(number_of_nodes))
        segmentation = np.linspace(0.0, 1.0, number_of_elements + 1)
        count = 1
        for point_Y in segmentation:
            for point_X in segmentation:
                X = 48.0 * point_X
                Y = ((1.0 - point_Y) * 44.0 + point_Y * 16.0) * point_X + 44.0 * point_Y
                mesh_file.write("{} {} {}\n".format(count, X, Y))
                count += 1
        nb_el = 2 * (number_of_elements * number_of_elements)
        mesh_file.write("**element\n{}\n".format(nb_el))
        count = 1
        cc_out = 0
        triangles = []
        for count_node in range(number_of_elements):
            for count_line in range(number_of_elements):
                if cc_out % 2 == 0:
                    # print("FALSE")
                    count_node_eff = count_node + 1
                    count_line_eff = count_line
                    p1 = count_node_eff + (count_line_eff * nb_nd)
                    p2 = count_node_eff + 1 + (count_line_eff * nb_nd)
                    p3 = count_node_eff + nb_nd + (count_line_eff * nb_nd)
                    mesh_file.write("{} c2d3 {} {} {}\n".format(count, p1, p2, p3))
                    triangles.append([p1 - 1, p2 - 1, p3 - 1])
                    count += 1
                    #
                    q1 = count_node_eff + 1 + (count_line_eff * nb_nd)
                    q2 = count_node_eff + nb_nd + 1 + (count_line_eff * nb_nd)
                    q3 = count_node_eff + nb_nd + (count_line_eff * nb_nd)
                    # q2 = count_node_eff + nb_nd + (count_line_eff * nb_nd)
                    # q3 = count_node_eff + nb_nd + 1 + (count_line_eff * nb_nd)
                    # mesh_file.write("{} c2d3 {} {} {}\n".format(count, q1, q3, q2))
                    mesh_file.write("{} c2d3 {} {} {}\n".format(count, q1, q2, q3))
                    # triangles.append([q1 - 1, q3 - 1, q2 - 1])
                    triangles.append([q1 - 1, q2 - 1, q3 - 1])
                    count += 1
                else:
                    # print("TRUE")
                    count_node_eff = count_node + 1
                    count_line_eff = count_line
                    p1 = count_node_eff + (count_line_eff * nb_nd)
                    p2 = count_node_eff + 1 + (count_line_eff * nb_nd)
                    p3 = count_node_eff + nb_nd + 1 + (count_line_eff * nb_nd)
                    mesh_file.write("{} c2d3 {} {} {}\n".format(count, p1, p2, p3))
                    triangles.append([p1 - 1, p2 - 1, p3 - 1])
                    count += 1
                    #
                    q3 = count_node_eff + nb_nd + (count_line_eff * nb_nd)
                    q2 = count_node_eff + (count_line_eff * nb_nd)
                    q1 = count_node_eff + nb_nd + 1 + (count_line_eff * nb_nd)
                    mesh_file.write("{} c2d3 {} {} {}\n".format(count, q1, q2, q3))
                    triangles.append([q1 - 1, q2 - 1, q3 - 1])
                    count += 1
                cc_out += 1
        eps = 0.001
        mesh_file.write("***group\n")
        mesh_file.write("**nset TOP\n")
        count = 1
        for point_Y in segmentation:
            for point_X in segmentation:
                X = 48.0 * point_X
                Y = ((1.0 - point_Y) * 44.0 + point_Y * 16.0) * point_X + 44.0 * point_Y
                if np.abs(Y - 44.0 - 16.0 / 48.0 * X) < eps:
                    mesh_file.write(" {}".format(count))
                count += 1
        mesh_file.write("\n")
        mesh_file.write("**nset BOTTOM\n")
        count = 1
        for point_Y in segmentation:
            for point_X in segmentation:
                X = 48.0 * point_X
                Y = ((1.0 - point_Y) * 44.0 + point_Y * 16.0) * point_X + 44.0 * point_Y
                if np.abs(Y - 0.0 - 44.0 / 48.0 * X) < eps:
                    mesh_file.write(" {}".format(count))
                count += 1
        mesh_file.write("\n")
        mesh_file.write("**nset LEFT\n")
        count = 1
        for point_Y in segmentation:
            for point_X in segmentation:
                X = 48.0 * point_X
                Y = ((1.0 - point_Y) * 44.0 + point_Y * 16.0) * point_X + 44.0 * point_Y
                if X == 0.0:
                    mesh_file.write(" {}".format(count))
                count += 1
        mesh_file.write("\n")
        mesh_file.write("**nset RIGHT\n")
        count = 1
        for point_Y in segmentation:
            for point_X in segmentation:
                X = 48.0 * point_X
                Y = ((1.0 - point_Y) * 44.0 + point_Y * 16.0) * point_X + 44.0 * point_Y
                if X == 48.0:
                    mesh_file.write(" {}".format(count))
                count += 1
        mesh_file.write("\n")
        mesh_file.write("**nset APPLI\n")
        count = 1
        for point_Y in segmentation:
            for point_X in segmentation:
                X = 48.0 * point_X
                Y = ((1.0 - point_Y) * 44.0 + point_Y * 16.0) * point_X + 44.0 * point_Y
                if X == 48.0 and np.abs(Y - 52.0) < 16.0 / (number_of_elements) + eps:
                    mesh_file.write(" {}".format(count))
                count += 1
        mesh_file.write("\n")
        mesh_file.write("***return")
    return triangles


number_of_elements = 6
face_polynomial_order = 1
cell_polynomial_order = 1
operator_type = "HDGs"
stabilization_parameter = 2.0 * mu

solve_2D_incompressible_problem(
    number_of_elements,
    face_polynomial_order,
    cell_polynomial_order,
    operator_type,
    stabilization_parameter,
)

# ------------------------------------------------------------------------------------------------------------------
# coef = 2.0
# tangent_matrix_lam = np.array(
#     [
#         [lam, lam, 0.0, 0.0],
#         [lam, lam, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0],
#     ]
# )
# tangent_matrix_mu = np.array(
#     [
#         [coef * mu, 0.0, 0.0, 0.0],
#         [0.0, coef * mu, 0.0, 0.0],
#         [0.0, 0.0, coef * mu, 0.0],
#         [0.0, 0.0, 0.0, coef * mu],
#     ]
# )
# tangent_matrix = tangent_matrix_lam + tangent_matrix_mu
# tangent_matrices = [tangent_matrix for i in range(len(cells))]