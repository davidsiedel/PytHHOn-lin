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
    for cell_vertices_connectivity_matrix, cell_connectivity_matrix in zip(
        cells_vertices_connectivity_matrix, cells_connectivity_matrix
    ):
        cell_vertices = vertices[cell_vertices_connectivity_matrix]
        cell = Cell(cell_vertices, cell_connectivity_matrix, integration_order)
        cells.append(cell)
    print("cells done")
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
    print("transfer done")
    total_system_size = number_of_faces * face_basis_k.basis_dimension * unknown.field_dimension
    global_matrix = np.zeros((total_system_size, total_system_size))
    global_vector = np.zeros((total_system_size,))
    print("global matrices done")
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
    double_lagrange = True
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
    # ------------------------------------------------------------------------------------------------------------------
    # Solving the global system for faces unknowns
    # ------------------------------------------------------------------------------------------------------------------
    print("inv : {}".format(np.linalg.inv(global_matrix_2) @ global_matrix_2))
    global_solution = np.linalg.solve(global_matrix_2, global_vector_2)
    print(global_matrix_2.shape)
    # global_matrix_2_sparse = csr_matrix(global_matrix_2)
    # global_matrix_2_sparse = dia_matrix(global_matrix_2)
    # global_solution = spsolve(global_matrix_2_sparse, global_vector_2)
    internal_forces = global_matrix_2[:total_system_size, :total_system_size] @ (global_solution[:total_system_size]).T
    print("res : {}".format(internal_forces - global_vector_2[:total_system_size]))
    # ------------------------------------------------------------------------------------------------------------------
    # Getting the number of vertices
    # ------------------------------------------------------------------------------------------------------------------
    number_of_quadrature_points = 0
    for cell in cells:
        number_of_quadrature_points_in_cell = cell.quadrature_points.shape[0]
        number_of_quadrature_points += number_of_quadrature_points_in_cell
    number_of_vertices = vertices.shape[0]
    # ------------------------------------------------------------------------------------------------------------------
    quadrature_points = np.zeros((number_of_quadrature_points, unknown.problem_dimension))
    quadrature_weights = np.zeros((number_of_quadrature_points,))
    unknowns_at_vertices = np.zeros((number_of_vertices, unknown.field_dimension))
    f_unknowns_at_vertices = np.zeros((number_of_vertices, unknown.field_dimension))
    unknowns_at_quadrature_points = np.zeros((number_of_quadrature_points, unknown.field_dimension))
    error_at_quadrature_points = np.zeros((number_of_quadrature_points, unknown.field_dimension))
    div_at_quadrature_points = np.zeros((number_of_quadrature_points,))
    stress_at_quadrature_points = []
    strain_at_quadrature_points = []
    vertices_weights = np.zeros((number_of_vertices,))
    f_vertices_weights = np.zeros((number_of_vertices,))
    # ==================================================================================================================
    # MGIS
    # ==================================================================================================================
    number_of_quadrature_points_in_mesh = 0
    for cell_index in cells_indices:
        local_cell = cells[cell_index]
        number_of_quadrature_points_in_mesh += len(local_cell.quadrature_points)
    # lib = "/Users/davidsiedel/Projects/PytHHOn3D/bhv/src/libBehaviour.dylib"
    # h = mgis_bv.Hypothesis.Tridimensional
    # b = mgis_bv.load(lib, "Elasticity", h)
    # # material data manager
    # m = mgis_bv.MaterialDataManager(b, number_of_quadrature_points_in_mesh)
    # m.s0.setMaterialProperty("YoungModulus", 1.124999981250001)
    # m.s0.setMaterialProperty("PoissonRatio", 0.499999999999999)
    # m.s1.setMaterialProperty("YoungModulus", 1.124999981250001)
    # m.s1.setMaterialProperty("PoissonRatio", 0.499999999999999)
    # T = 293.15 * np.ones(number_of_quadrature_points_in_mesh)
    # Ts = mgis_bv.MaterialStateManagerStorageMode.ExternalStorage
    # mgis_bv.setExternalStateVariable(m.s0, "Temperature", T, Ts)
    # mgis_bv.setExternalStateVariable(m.s1, "Temperature", T, Ts)
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
                f_unknowns_at_vertices[l0:l1, direction] += np.sum(vertex_value_vector)
            f_vertices_weights[l0:l1] += 1.0
    # ==================================================================================================================
    # ==================================================================================================================
    x_cell_list, x_faces_list = [], []
    distance = 10e9
    pA_cell_index = 0
    point_A = (48.0, 52.0)
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
        m_error = Integration.get_cell_mass_matrix_in_cell(local_cell, cell_basis_l, cell_basis_l)
        m_error_inv = np.linalg.inv(m_error)
        error_vector = np.zeros((len(x_cell),))
        for direction in range(unknown.field_dimension):
            l0 = direction * cell_basis_l.basis_dimension
            l1 = (direction + 1) * cell_basis_l.basis_dimension
            anal_sol = u[direction]
            anal_vector = Integration.get_cell_load_vector_in_cell(local_cell, cell_basis_l, anal_sol)
            error_vector[l0:l1] = np.abs(m_error_inv @ anal_vector - x_cell[l0:l1])
        # operators[cell_index]
        # --------------------------------------------------------------------------------------------------------------
        cell_vertices_connectivity_matrix = cells_vertices_connectivity_matrix[cell_index]
        local_vertices = vertices[cell_vertices_connectivity_matrix]
        # --------------------------------------------------------------------------------------------------------------
        for i, vertex in enumerate(local_vertices):
            global_index = cell_vertices_connectivity_matrix[i]
            v = cell_basis_l.get_phi_vector(vertex, local_cell.centroid, local_cell.diameter)
            for direction in range(unknown.field_dimension):
                l0 = direction * cell_basis_l.basis_dimension
                l1 = (direction + 1) * cell_basis_l.basis_dimension
                vertex_value_vector = v * x_cell[l0:l1]
                unknowns_at_vertices[global_index, direction] += np.sum(vertex_value_vector)
            vertices_weights[global_index] += 1.0
        # --------------------
        local_distance = np.sqrt(
            (local_cell.centroid[0] - point_A[0]) ** 2 + (local_cell.centroid[1] - point_A[1]) ** 2
        )
        if local_distance < distance:
            distance = local_distance
            pA_cell_index = cell_index
        # --------------------------------------------------------------------------------------------------------------
        # DIVERGENCE draft
        # --------------------------------------------------------------------------------------------------------------
        grad_op = operators[cell_index].local_gradient_operator
        grad_vect = grad_op @ x_element
        for i, quadrature_point in enumerate(local_cell.quadrature_points):
            v_k = cell_basis_k.get_phi_vector(quadrature_point, local_cell.centroid, local_cell.diameter)
            v_l = cell_basis_l.get_phi_vector(quadrature_point, local_cell.centroid, local_cell.diameter)
            stress = np.zeros((4,))
            strain = np.zeros((4,))
            # strain = np.zeros(2, 2)
            for direction in range(unknown.field_dimension):
                # ------------------------------------------------------------------------------------------------------
                l0_l = direction * cell_basis_l.basis_dimension
                l1_l = (direction + 1) * cell_basis_l.basis_dimension
                # ------------------------------------------------------------------------------------------------------
                l0_k = direction * cell_basis_k.basis_dimension
                l1_k = (direction + 1) * cell_basis_k.basis_dimension
                # ------------------------------------------------------------------------------------------------------
                # div_cell = v_k * grad_vect[l0_k:l1_k]
                # xcell_vector = v_l * x_cell[l0_l:l1_l]
                # ercell_vector = v_l * error_vector[l0_l:l1_l]
                # l0 = quadrature_point_count
                # l1 = quadrature_point_count + 1
                div_at_quadrature_points[quadrature_point_count] += v_k.T @ grad_vect[l0_k:l1_k]
                unknowns_at_quadrature_points[quadrature_point_count, direction] += v_l.T @ x_cell[l0_l:l1_l]
                error_at_quadrature_points[quadrature_point_count, direction] += v_l.T @ error_vector[l0_l:l1_l]
                # ----------
                strain[direction] += v_k.T @ grad_vect[l0_k:l1_k]
            # --------------
            l0_k = 2 * cell_basis_k.basis_dimension
            l1_k = (2 + 1) * cell_basis_k.basis_dimension
            strain[3] += np.sqrt(2.0) * v_k.T @ grad_vect[l0_k:l1_k]
            # --------------
            l0_k = 3 * cell_basis_k.basis_dimension
            l1_k = (3 + 1) * cell_basis_k.basis_dimension
            # strain[3] += np.sqrt(2) * v_k.T @ grad_vect[l0_k:l1_k]
            strain[2] += 0.0
            # -------
            stress = np.array(
                [
                    2.0 * mu * strain[0] + lam * (strain[0] + strain[1]),
                    2.0 * mu * strain[1] + lam * (strain[0] + strain[1]),
                    # 0.49999997500000126 * (2.0 * mu + lam) * (strain[0] + strain[1]),
                    lam * (strain[0] + strain[1]),
                    # np.sqrt(2.0) * 2.0 * mu * strain[3],
                    2.0 * mu * strain[3],
                ]
            )
            # ----------------------------------------------------------------------------------------------------------
            lib = "/Users/davidsiedel/Projects/PytHHOn3D/bhv/src/libBehaviour.dylib"
            h = mgis_bv.Hypothesis.Tridimensional
            h = mgis_bv.Hypothesis.PlaneStrain
            # h = mgis_bv.Hypothesis.PlaneStress
            b = mgis_bv.load(lib, "Elasticity", h)
            # material data manager
            m = mgis_bv.MaterialDataManager(b, 1)
            # E = 1.124999981250001
            # nu = 0.49999997500000126
            m.s0.setMaterialProperty("YoungModulus", 1.12499998125000100)
            m.s0.setMaterialProperty("PoissonRatio", 0.49999997500000126)
            m.s1.setMaterialProperty("YoungModulus", 1.12499998125000100)
            m.s1.setMaterialProperty("PoissonRatio", 0.49999997500000126)
            T = 293.15 * np.ones(1)
            Ts = mgis_bv.MaterialStateManagerStorageMode.ExternalStorage
            mgis_bv.setExternalStateVariable(m.s0, "Temperature", T, Ts)
            mgis_bv.setExternalStateVariable(m.s1, "Temperature", T, Ts)
            it = mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator
            # it = mgis_bv.IntegrationType.IntegrationWithTangentOperator
            m.s0.gradients[0] = np.zeros((4,))
            m.s1.gradients[0] += strain
            m.s0.thermodynamic_forces[0] = np.zeros((4,))
            # for i in range(0, 1):
            # it = mgis_bv.IntegrationType.IntegrationWithTangentOperator
            dt = 0.0
            # for i in range(2):
            # mgis_bv.integrate(m, it, dt, 0, m.n)
            # print("K : \n{}".format(m.K))
            # mgis_bv.update(m)
            # print("diff : {}".format(m.s1.thermodynamic_forces[0] - stress))
            # print(m.s1.thermodynamic_forces[0].shape)
            # for p in range(0, number_of_quadrature_points_in_mesh - 1):
            #     m.s1.gradients[p][0] += 0.0
            #     print(m.s1.thermodynamic_forces[p])
            # ----------------------------------------------------------------------------------------------------------
            point_A = (48.0, 52.0)
            # if np.sqrt((quadrature_point[0] - point_A[0]) ** 2 + (vertex[1] - point_A[1]) ** 2) < 0.001:
            #     sig_p_A = (1.0 / 3.0) * (2.0 * mu + lam) * div_at_quadrature_points[i]
            #     div_p_A = lam * div_at_quadrature_points[i]
            #     ux_A = unknowns_at_quadrature_points[i, 0]
            #     uy_A = unknowns_at_quadrature_points[i, 1]
            #     print("ux_A : {}".format(ux_A))
            #     print("uy_A : {}".format(uy_A))
            #     print("sig_p_A : {}".format(sig_p_A))
            #     print("div_p_A : {}".format(div_p_A))
            quadrature_points[quadrature_point_count] += quadrature_point
            quadrature_weights[quadrature_point_count] += local_cell.quadrature_weights[i]
            quadrature_point_count += 1
    # ==================================================================================================================
    # MFRONT MGIS PART
    # ==================================================================================================================
    pA_cell = cells[pA_cell_index]
    pA_xcell = x_cell_list[pA_cell_index]
    pA_xfaces = x_faces_list[pA_cell_index]
    pA_xelement = np.zeros((len(pA_xcell) + len(pA_xfaces)))
    pA_xelement[: cell_basis_l.basis_dimension * unknown.field_dimension] += pA_xcell
    pA_xelement[cell_basis_l.basis_dimension * unknown.field_dimension :] += pA_xfaces
    pA_grad = operators[pA_cell_index].local_gradient_operator
    pA_gradvec = pA_grad @ pA_xelement
    v_k = cell_basis_k.get_phi_vector(point_A, pA_cell.centroid, pA_cell.diameter)
    v_l = cell_basis_l.get_phi_vector(point_A, pA_cell.centroid, pA_cell.diameter)
    pA_div = 0.0
    pA_u = np.array([0.0, 0.0])
    for direction in range(unknown.field_dimension):
        # ------------------------------------------------------------------------------------------------------
        l0_l = direction * cell_basis_l.basis_dimension
        l1_l = (direction + 1) * cell_basis_l.basis_dimension
        # ------------------------------------------------------------------------------------------------------
        l0_k = direction * cell_basis_k.basis_dimension
        l1_k = (direction + 1) * cell_basis_k.basis_dimension
        # ------------------------------------------------------------------------------------------------------
        pA_div_cell = v_k * pA_gradvec[l0_k:l1_k]
        pA_xcell_vector = v_l * pA_xcell[l0_l:l1_l]
        pA_div += np.sum(pA_div_cell)
        pA_u[direction] += np.sum(pA_xcell_vector)
    sig_p_A = (1.0 / 3.0) * (2.0 * mu + lam) * pA_div
    div_p_A = lam * pA_div
    ux_A = pA_u[0]
    uy_A = pA_u[1]
    print(vertices[cells_vertices_connectivity_matrix[pA_cell_index]])
    print("sig_p_A = {}".format(sig_p_A))
    print("div_p_A = {}".format(div_p_A))
    print("ux_A = {}".format(ux_A))
    print("uy_A = {}".format(uy_A))

    # grad_op = operators[cell_index].local_gradient_operator
    # grad_vect = grad_op @ x_element
    # for i, quadrature_point in enumerate(local_cell.quadrature_points):
    #     v = cell_basis_k.get_phi_vector(quadrature_point, local_cell.centroid, local_cell.diameter)
    #     for direction in range(unknown.field_dimension):
    #         l0 = direction * cell_basis_k.basis_dimension
    #         l1 = (direction + 1) * cell_basis_k.basis_dimension
    #         vertex_value_vector = v * grad_vect[l0:l1]
    #         l0 = quadrature_point_count
    #         l1 = quadrature_point_count + 1
    #         div_at_quadrature_points[l0:l1] += np.sum(vertex_value_vector)
    # # --------------------------------------------------------------------------------------------------------------
    # for i, quadrature_point in enumerate(local_cell.quadrature_points):
    #     v = cell_basis_l.get_phi_vector(quadrature_point, local_cell.centroid, local_cell.diameter)
    #     for direction in range(unknown.field_dimension):
    #         l0 = direction * cell_basis_l.basis_dimension
    #         l1 = (direction + 1) * cell_basis_l.basis_dimension
    #         vertex_value_vector = v * x_cell[l0:l1]
    #         vertex_error_vector = v * error_vector[l0:l1]
    #         l0 = quadrature_point_count
    #         l1 = quadrature_point_count + 1
    #         unknowns_at_quadrature_points[l0:l1, direction] += np.sum(vertex_value_vector)
    #         error_at_quadrature_points[l0:l1, direction] += np.sum(vertex_error_vector)
    #     quadrature_points[quadrature_point_count] += quadrature_point
    #     quadrature_weights[quadrature_point_count] += local_cell.quadrature_weights[i]
    #     quadrature_point_count += 1
    # ------------------------------------------------------------------------------------------------------------------
    # Global matrix
    # ------------------------------------------------------------------------------------------------------------------
    for direction in range(unknown.field_dimension):
        unknowns_at_vertices[:, direction] = unknowns_at_vertices[:, direction] / vertices_weights
        f_unknowns_at_vertices[:, direction] = f_unknowns_at_vertices[:, direction] / f_vertices_weights
        # unknowns_at_vertices = unknowns_at_vertices / vertices_weights
        # f_unknowns_at_vertices = f_unknowns_at_vertices / f_vertices_weights
    return (
        (vertices, unknowns_at_vertices),
        (
            quadrature_points,
            unknowns_at_quadrature_points,
            error_at_quadrature_points,
            div_at_quadrature_points,
            quadrature_weights,
        ),
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
    stabilization_parameter: float,
):
    field_dimension = 2
    mesh_file = os.path.join(source, "meshes/2D/c2d3_{}_cooke.geof".format(number_of_elements))
    triangles = create_plot_cooke(mesh_file, number_of_elements)
    # ------------------------------------------------------------------------------------------------------------------
    pressure_left = [None, None]
    pressure_right = [None, lambda x: 1.0 / 16.0]
    pressure_top = [None, None]
    pressure_bottom = [None, None]
    pressure_appli = [None, None]
    # ------------------------------------------------------------------------------------------------------------------
    displacement_left = [lambda x: 0.0, lambda x: 0.0]
    displacement_right = [None, None]
    displacement_top = [None, None]
    displacement_bottom = [None, None]
    displacement_appli = [None, None]
    # ------------------------------------------------------------------------------------------------------------------
    load = [lambda x: 0.0, lambda x: 0.0]
    # ------------------------------------------------------------------------------------------------------------------
    boundary_conditions = {
        "RIGHT": (displacement_right, pressure_right),
        "LEFT": (displacement_left, pressure_left),
        "TOP": (displacement_top, pressure_top),
        "BOTTOM": (displacement_bottom, pressure_bottom),
        "APPLI": (displacement_appli, pressure_appli),
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
    coef = 2.0
    tangent_matrix_lam = np.array(
        [
            [lam, lam, 0.0, 0.0],
            [lam, lam, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    tangent_matrix_mu = np.array(
        [
            [coef * mu, 0.0, 0.0, 0.0],
            [0.0, coef * mu, 0.0, 0.0],
            [0.0, 0.0, coef * mu, 0.0],
            [0.0, 0.0, 0.0, coef * mu],
        ]
    )
    tangent_matrix = tangent_matrix_lam + tangent_matrix_mu
    tangent_matrices = [tangent_matrix for i in range(len(cells))]
    # ------------------------------------------------------------------------------------------------------------------
    (
        (vertices, unknowns_at_vertices),
        (
            quadrature_points,
            unknowns_at_quadrature_points,
            error_at_quadrature_points,
            div_at_quadrature_points,
            quadrature_weights,
        ),
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
                    print("FALSE")
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
                    print("TRUE")
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
    number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter
)
