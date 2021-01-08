import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os
from tests.test_error.analysis_1d import *

number_of_elements = 100
face_polynomial_order = 1
cell_polynomial_order = 2
operator_type = "HDG"
stabilization_parameter = 1.0

plot_1D_solution(
    number_of_elements, face_polynomial_order, cell_polynomial_order, operator_type, stabilization_parameter
)
