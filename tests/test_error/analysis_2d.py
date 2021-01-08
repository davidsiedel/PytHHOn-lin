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


def solve_2D_poisson_problem(
    number_of_elements: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    stabilization_parameter: int,
):
    # field_dimension = 1
    # mesh_file = os.path.join(source, "meshes/2D/c2d3_{}.geof".format(number_of_elements))
    number_of_nodes = number_of_elements + 1
    field_dimension = 1
    mesh_file = os.path.join(source, "meshes/2D/c2d3_{}.geof".format(number_of_elements))
    # if not os.path.exists(mesh_file):
    #     create_plot(mesh_file, number_of_elements)
    create_plot(mesh_file, number_of_elements)
    # ------------------------------------------------------------------------------------------------------------------
    pressure_left = [None]
    pressure_right = [None]
    pressure_top = [None]
    pressure_bottom = [None]
    # ------------------------------------------------------------------------------------------------------------------
    displacement_left = [None]
    displacement_right = [None]
    displacement_top = [lambda x: 0.0]
    displacement_bottom = [lambda x: 0.0]
    load = [lambda x: np.sin(2.0 * np.pi * x[1])]
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
    d = unknown.problem_dimension
    tangent_matrices = [np.eye(2) for i in range(len(cells))]
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
    return (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, error_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    )


def solve_2D_incompressible_problem(
    number_of_elements: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    stabilization_parameter: int,
):
    field_dimension = 2
    lam, mu = 1000.0, 1.0
    mesh_file = os.path.join(source, "meshes/2D/c2d3_{}.geof".format(number_of_elements))
    # ------------------------------------------------------------------------------------------------------------------
    pressure_left = [None, None]
    pressure_right = [None, None]
    pressure_top = [None, None]
    pressure_bottom = [None, None]
    # ------------------------------------------------------------------------------------------------------------------
    anal_sol_x = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + 1.0 / (2.0 * lam) * x[0]
    anal_sol_y = lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]) + 1.0 / (2.0 * lam) * x[1]
    # ------------------------------------------------------------------------------------------------------------------
    dirichlet_x_bottom = lambda s: anal_sol_x([s, 0.0])
    dirichlet_y_bottom = lambda s: anal_sol_y([s, 0.0])
    # ------------------------------------------------------------------------------------------------------------------
    dirichlet_x_top = lambda s: anal_sol_x([s, 1.0])
    dirichlet_y_top = lambda s: anal_sol_y([s, 1.0])
    # ------------------------------------------------------------------------------------------------------------------
    dirichlet_x_left = lambda s: anal_sol_x([0.0, s])
    dirichlet_y_left = lambda s: anal_sol_y([0.0, s])
    # ------------------------------------------------------------------------------------------------------------------
    dirichlet_x_right = lambda s: anal_sol_x([1.0, s])
    dirichlet_y_right = lambda s: anal_sol_y([1.0, s])
    # ------------------------------------------------------------------------------------------------------------------
    load_x = lambda x: 2.0 * (np.pi ** 2) * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    load_y = lambda x: 2.0 * (np.pi ** 2) * np.cos(np.pi * x[0]) * np.cos(np.pi * x[1])
    displacement_left = [dirichlet_x_left, dirichlet_y_left]
    displacement_right = [dirichlet_x_right, dirichlet_y_right]
    displacement_top = [dirichlet_x_top, dirichlet_y_top]
    displacement_bottom = [dirichlet_x_bottom, dirichlet_y_bottom]
    load = [load_x, load_y]
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
    d = unknown.problem_dimension
    tangent_matrices = [np.eye(2) for i in range(len(cells))]
    tangent_matrix = np.array(
        [
            [lam + 2.0 * mu, lam, 0.0, 0.0],
            [lam, lam + 2.0 * mu, 0.0, 0.0],
            [0.0, 0.0, 2.0 * mu, 0.0],
            [0.0, 0.0, 0.0, 2.0 * mu],
        ]
    )
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
    return (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, error_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    )


def compute_2D_error(
    number_of_elements: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    stabilization_parameter: int,
):
    (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, error_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    ) = solve_2D_poisson_problem(
        number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter
    )
    anal_sol = lambda x: -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * x[0]))
    load = lambda x: (np.sin(2.0 * np.pi * x[0]))
    error = 0.0
    error_plus = 0.0
    for i in range(len(quadrature_points)):
        error += np.abs((anal_sol(quadrature_points[i]) - unknowns_at_quadrature_points[0][i]) * quadrature_weights[i])
        error += error_at_quadrature_points[0][i] * quadrature_weights[i]
        # error += ((anal_sol(quadrature_points[i]) - unknowns_at_quadrature_points[0][i]) * quadrature_weights[i]) ** 2
        error_plus += (anal_sol(quadrature_points[i]) * quadrature_weights[i]) ** 2 + (
            load(quadrature_points[i]) * quadrature_weights[i]
        ) ** 2
    print("error_plus : {}, k = {}, l = {}".format(error_plus, face_polynomial_order, cell_polynomial_order))
    return error, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter


def plot_2D_error(
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
    # h_0 = 1.0
    # e_0 = 1.0
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


def plot_2D_error_map(
    number_of_elements: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    stabilization_parameter: int,
    fig,
    ax,
):
    (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, error_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    ) = solve_2D_poisson_problem(
        number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter
    )
    vmax = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * 0.75))
    x, y = quadrature_points.T
    # print(unknowns_at_quadrature_points[0].shape)
    # print(quad_p.shape)
    error = unknowns_at_quadrature_points[0] - np.array(
        [-(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * quad_p)) for quad_p in y]
    )
    error = np.abs(error) / vmax * 100.0
    data = ax.tricontourf(x, y, error, cmap=cm.binary)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("map of the domain $\Omega$")
    ax.set_title("Error")
    cbar = fig.colorbar(data, ax=ax)
    cbar.set_label("Relative error (percent)", rotation=270, labelpad=15.0)
    return


def plot_2D_incompressible_error_map(
    number_of_elements: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    stabilization_parameter: int,
    fig,
    ax,
):
    (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, error_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    ) = solve_2D_incompressible_problem(
        number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter
    )
    lam, mu = 1000.0, 1.0
    anal_sol_x = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + 1.0 / (2.0 * lam) * x[0]
    anal_sol_y = lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]) + 1.0 / (2.0 * lam) * x[1]
    nb = 500
    x = np.linspace(0.0, 1.0, nb)
    pts = []
    for i in range(nb):
        for j in range(nb):
            pts.append(np.array([x[i], x[j]]))
    vmax = max([anal_sol_x(pt) for pt in pts])
    x, y = quadrature_points.T
    error = unknowns_at_quadrature_points[0] - np.array([anal_sol_x(quad_p) for quad_p in quadrature_points])
    error = np.abs(error) / vmax * 100.0
    data = ax.tricontourf(x, y, error, cmap=cm.binary)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("map of the domain $\Omega$")
    ax.set_title("Error")
    cbar = fig.colorbar(data, ax=ax)
    cbar.set_label("Relative error (percent)", rotation=270, labelpad=15.0)
    return


def plot_2D_solution_map(
    problem: Callable,
    number_of_elements: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    stabilization_parameter: int,
    fig,
    ax,
):
    (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, error_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    ) = problem(
        number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter
    )
    vmax = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * 0.75))
    vmin = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * 0.25))
    x, y = quadrature_points.T
    # data = ax.tricontourf(x, y, unknowns_at_quadrature_points[0], cmap=cm.binary)
    data = ax.tricontourf(x, y, unknowns_at_quadrature_points[0], vmin=vmin, vmax=vmax, cmap=cm.binary)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("map of the domain $\Omega$")
    ax.set_title(
        "{}, $k = {}, l = {}$, stab $= {}$".format(
            operator_type, face_polynomial_order, cell_polynomial_order, stabilization_parameter
        )
    )
    cbar = fig.colorbar(data, ax=ax)
    cbar.set_label("{} solution".format(operator_type), rotation=270, labelpad=15.0)
    return


def plot_2D_incompressible_solution_map(
    problem: Callable,
    number_of_elements: int,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    stabilization_parameter: int,
    fig,
    ax,
):
    (
        (vertices, unknowns_at_vertices),
        (quadrature_points, unknowns_at_quadrature_points, error_at_quadrature_points, quadrature_weights),
        (vertices, f_unknowns_at_vertices),
        (x_cell_list, x_faces_list),
    ) = problem(
        number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter
    )
    # vmax = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * 0.75))
    # vmin = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * 0.25))
    x, y = quadrature_points.T
    data = ax.tricontourf(x, y, unknowns_at_quadrature_points[0], cmap=cm.binary)
    # data = ax.tricontourf(x, y, unknowns_at_quadrature_points[1], cmap=cm.binary)
    # data = ax.tricontourf(x, y, unknowns_at_quadrature_points[0], vmin=vmin, vmax=vmax, cmap=cm.binary)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("map of the domain $\Omega$")
    ax.set_title(
        "{}, $k = {}, l = {}$, stab $= {}$".format(
            operator_type, face_polynomial_order, cell_polynomial_order, stabilization_parameter
        )
    )
    cbar = fig.colorbar(data, ax=ax)
    cbar.set_label("{} solution".format(operator_type), rotation=270, labelpad=15.0)
    return


def plot_2D_analytical_map(fig, ax):
    x = np.linspace(0.0, 1.0, 1000)
    y = np.linspace(0.0, 1.0, 1000)
    X, Y = np.meshgrid(x, y)
    Z = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * X))
    Z = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * Y))
    vmax = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * 0.75))
    vmin = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * 0.25))
    col_levels = 10
    CS = ax.contourf(X, Y, Z, col_levels, vmin=vmin, vmax=vmax, cmap=cm.binary)
    m = plt.cm.ScalarMappable(cmap=cm.binary)
    m.set_array(Z)
    m.set_clim(vmin, vmax)
    cbar = fig.colorbar(m, boundaries=np.linspace(vmin, vmax, col_levels), ax=ax)
    cbar.set_label("Analytical solution", rotation=270, labelpad=15.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    ax.set_xlabel("map of the domain $\Omega$")
    ax.set_title("Analytical Solution")
    return


def plot_2D_incompressible_analytical_map(fig, ax):
    x = np.linspace(0.0, 1.0, 1000)
    y = np.linspace(0.0, 1.0, 1000)
    X, Y = np.meshgrid(x, y)
    lam, mu = 1000.0, 1.0
    anal_sol_x = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) + 1.0 / (2.0 * lam) * x[0]
    anal_sol_y = lambda x: np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]) + 1.0 / (2.0 * lam) * x[1]
    # Z = -(1.0 / (2.0 * np.pi) ** 2) * (np.sin(2.0 * np.pi * X))
    Z = np.sin(np.pi * X) * np.sin(np.pi * Y) + 1.0 / (2.0 * lam) * X
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
    CS = ax.contourf(X, Y, Z, col_levels, vmin=vmin, vmax=vmax, cmap=cm.binary)
    m = plt.cm.ScalarMappable(cmap=cm.binary)
    m.set_array(Z)
    m.set_clim(vmin, vmax)
    cbar = fig.colorbar(m, boundaries=np.linspace(vmin, vmax, col_levels), ax=ax)
    cbar.set_label("Analytical solution", rotation=270, labelpad=15.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("map of the domain $\Omega$")
    ax.set_title("Analytical Solution")
    return


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

