import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os
from tests.test_error.analysis_2d import *

number_of_elements = 30
face_polynomial_order = 2
cell_polynomial_order = 2
operator_type = "HDG"
stabilization_parameter = 1.0

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

plot_2D_error_map(
    number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter, fig, ax0
)
plot_2D_solution_map(
    solve_2D_poisson_problem,
    number_of_elements,
    face_polynomial_order,
    cell_polynomial_order,
    operator_type,
    stabilization_parameter,
    fig,
    ax1,
)
plot_2D_analytical_map(fig, ax2)

plt.show()
