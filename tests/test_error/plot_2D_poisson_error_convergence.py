import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os

# from random import randrange
import string
import random
from tests.test_error.analysis_2d import *

from tests.context import source

operator_type = "HHO"
stabilization_parameter = 1.0
expected_convergence_rate = 2

fig, ax = plt.subplots()
orders = [(1, 0), (2, 1)]
orders = [(2, 1), (3, 2)]
# orders = [(1, 1), (2, 2)]
# orders = [(1, 2), (2, 3)]
# orders = [(1, 1), (2, 2), (3, 3)]
# orders = [(1, 2), (2, 3), (3, 4)]
colors = [(flt, flt, flt) for flt in np.linspace(0.0, 0.5, len(orders), endpoint=True)]

for (face_polynomial_order, cell_polynomial_order), color in zip(orders, colors):
    # num_list = range(5, 35, 5)
    num_list = [5, 10, 15]
    num_list = [2, 3]
    num_list = range(1, 11)
    num_list = range(4, 11)
    h_list = [1 / float(num) for num in num_list]
    e_list = []
    for number_of_elements in num_list:
        error, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter = compute_2D_error(
            number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter
        )
        e_list.append(error)
    # if face_polynomial_order == 1 and cell_polynomial_order == 1:
    #     expected_convergence_rate = 1
    # elif face_polynomial_order == 1 and cell_polynomial_order == 2:
    #     expected_convergence_rate = 2
    plot_2D_error(h_list, e_list, face_polynomial_order, cell_polynomial_order, expected_convergence_rate, ax, color)

# ax.set_xlabel("$\log(h/h_0)$ with $h_0 = 1/5$")
ax.set_xlabel("$\log(h/h_0)$")
ax.set_ylabel("logartithmic $L^2$-error")
ax.set_ylim(-4.0, 0.5)
ax.set_title("{}, stab $= {}$".format(operator_type, stabilization_parameter))
plt.legend()
plt.grid(True)
serial = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
image_file_path = os.path.join(
    source, "plots/2D_{}_L2error_k{}convergence_{}.png".format(operator_type, expected_convergence_rate, serial)
)
while os.path.exists(image_file_path):
    serial = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
    image_file_path = os.path.join(
        source, "plots/2D_{}_L2error_k{}convergence_{}.png".format(operator_type, expected_convergence_rate, serial,),
    )
plt.savefig(image_file_path, dpi=fig.dpi)
