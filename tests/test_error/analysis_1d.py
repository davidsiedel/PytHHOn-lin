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

from pythhon3d import build, solve

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 12})
rc("text", usetex=True)


def solve_1D_poisson_problem(
    number_of_elements: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    stabilization_parameter: int,
):
    number_of_nodes = number_of_elements + 1
    field_dimension = 1
    mesh_file = os.path.join(source, "meshes/1D/c1d2_{}.geof".format(number_of_elements))
    if not os.path.exists(mesh_file):
        create_plot(mesh_file, number_of_elements)
    pressure_left = [None]
    pressure_right = [None]
    #
    displacement_left = [lambda x: 0.0]
    displacement_right = [lambda x: 0.0]
    #
    load = [lambda x: np.sin(2.0 * np.pi * x[0])]
    #
    boundary_conditions = {
        "RIGHT": (displacement_right, pressure_right),
        "LEFT": (displacement_left, pressure_left),
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
        # nsets,
        nsets_faces,
        cell_basis_l,
        cell_basis_k,
        face_basis_k,
        unknown,
    ) = build(mesh_file, field_dimension, face_polynomial_order, cell_polynomial_order, operator_type)
    # ------------------------------------------------------------------------------------------------------------------
    d = unknown.problem_dimension
    tangent_matrices = [np.eye(d ** 2) for i in range(len(cells))]
    # ------------------------------------------------------------------------------------------------------------------
    (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, quadrature_weights),
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
        # nsets,
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
    return (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    )


def compute_1D_error(
    number_of_elements: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    stabilization_parameter: int,
):
    (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    ) = solve_1D_poisson_problem(
        number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter
    )
    anal_sol = lambda x: -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * x))
    quadrature_points, unknowns_at_quadrature_points, quadrature_weights
    error = 0.0
    for i in range(len(quadrature_points)):
        error += np.sqrt(
            (quadrature_weights[i] * (unknowns_at_quadrature_points[0][i] - anal_sol(quadrature_points[i][0])) ** 2)
        )
        # error += (anal_sol(quadrature_points[i]) - unknowns_at_quadrature_points[0][i]) * quadrature_weights[i]
    print(error)
    return error, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter


def plot_1D_error(
    h_list: List,
    e_list: List,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    expected_convergence_rate: int,
    ax,
    color,
):
    h_0 = h_list[0]
    e_0 = e_list[0]
    ax.plot(
        [np.log(h_i / h_0) for h_i in h_list],
        [np.log(e_i / e_0) for e_i in e_list],
        label="logartithmic $L^2$-error for $k = {}, l = {}$".format(face_polynomial_order, cell_polynomial_order),
        color=color,
    )
    ax.plot(
        [np.log((h_i / h_0)) for h_i in h_list],
        [np.log((h_i / h_0) ** (face_polynomial_order + expected_convergence_rate)) for h_i in h_list],
        label="${}\log(h/h_0)$".format(face_polynomial_order + expected_convergence_rate),
        color=color,
        dashes=[2, 2],
    )
    ax.scatter(
        [np.log(h_i / h_0) for h_i in h_list], [np.log(e_i / e_0) for e_i in e_list], color=color,
    )
    return


def plot_1D_solution(
    number_of_elements: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    stabilization_parameter: int,
):
    (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    ) = solve_1D_poisson_problem(
        number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter
    )
    # ------------------------------------------------------------------------------------------------------------------
    analytical_segmentation = np.linspace(0.0, 1.0, 1000, endpoint=True)
    geometrical_segmentation = np.linspace(0.0, 1.0, number_of_elements, endpoint=True)
    # ------------------------------------------------------------------------------------------------------------------
    cell_sol = lambda x, x_T, v_T, u: np.sum([u[i] * ((x - x_T) / v_T) ** i for i in range(cell_polynomial_order + 1)])
    # ------------------------------------------------------------------------------------------------------------------
    anal_sol = lambda x: -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * x))
    fig, ax = plt.subplots()
    x_anal = analytical_segmentation
    y_anal = [anal_sol(xi) for xi in x_anal]
    ax.plot(x_anal, y_anal, color="black", linewidth=1, dashes=[2, 2])
    # ------------------------------------------------------------------------------------------------------------------
    for el_index in range(len(geometrical_segmentation) - 1):
        b_left = geometrical_segmentation[el_index]
        b_right = geometrical_segmentation[el_index + 1]
        # --------------------------------------------------------------------------------------------------------------
        n_left = int(1000.0 * b_left)
        n_right = int(1000.0 * b_right)
        # --------------------------------------------------------------------------------------------------------------
        if n_left > 1000:
            n_left = 1000
        if n_left < 0:
            n_left = 0
        # --------------------------------------------------------------------------------------------------------------
        if n_right > 1000:
            n_right = 1000
        if n_right < 0:
            n_right = 0
        # --------------------------------------------------------------------------------------------------------------
        x = analytical_segmentation[n_left:n_right]
        y = []
        for xi in x:
            v_T = b_right - b_left
            x_T = el_index * v_T + 0.5 * v_T
            yi = cell_sol(xi, x_T, v_T, x_cell_list[el_index])
            y.append(yi)
        ax.plot(x, y, color="black")
        # --------------------------------------------------------------------------------------------------------------
        f_unknowns_at_vertices
        ax.scatter(b_left, x_faces_list[el_index][0], color="grey")
        ax.scatter(b_right, x_faces_list[el_index][1], color="grey")
    # ------------------------------------------------------------------------------------------------------------------
    ax.set_xlabel("Position on the mesh $\Omega = [0,1]$")
    ax.set_ylabel("{} and analytical solutions".format(operator_type))
    ax.set_title(
        "{} Solution, $k = {}, l = {}$, stab $= {}$".format(
            operator_type, face_polynomial_order, cell_polynomial_order, stabilization_parameter
        )
    )
    cell_legend = mlines.Line2D(
        [],
        [],
        color="black",
        marker=None,
        markersize=6,
        linewidth=1.0,
        alpha=1.0,
        dashes=(None, None),
        label="cell unknowns",
    )
    face_legend = mlines.Line2D(
        [],
        [],
        color="grey",
        marker="o",
        markersize=6,
        linewidth=0.0,
        alpha=1.0,
        dashes=(None, None),
        label="face unknowns",
    )
    anal_legend = mlines.Line2D(
        [],
        [],
        color="black",
        marker=None,
        markersize=6,
        linewidth=1.0,
        alpha=1.0,
        dashes=(2, 2),
        label="analytical solution",
    )
    plt.legend(handles=[cell_legend, face_legend, anal_legend])
    plt.show()
    return


def create_plot(file_path: str, number_of_elements):
    with open(file_path, "w") as mesh_file:
        number_of_nodes = number_of_elements + 1
        mesh_file.write("***geometry\n**node\n{} 1\n".format(number_of_nodes))
        segmentation = np.linspace(0.0, 1.0, number_of_nodes)
        for i, point in enumerate(segmentation):
            mesh_file.write("{} {}\n".format(i + 1, point))
        mesh_file.write("**element\n{}\n".format(number_of_nodes - 1))
        for i, point in enumerate(segmentation[:-1]):
            mesh_file.write("{} c1d2 {} {}\n".format(i + 1, i + 1, i + 2))
        mesh_file.write("***group\n**nset LEFT\n1\n**nset RIGHT\n{}\n***return".format(number_of_nodes))
    return

