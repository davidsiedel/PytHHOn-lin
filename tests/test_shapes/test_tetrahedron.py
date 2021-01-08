from tests import context
import pytest
import numpy as np
from scipy import integrate
from shapes.tetrahedron import Tetrahedron


def test_tetrahedron_centroid():
    tetrahedron = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    polynomial_order = 1
    t = Tetrahedron(tetrahedron, polynomial_order)
    expected_centroid = np.array([1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0])
    assert (np.abs(t.centroid - expected_centroid) < np.full((3,), 1.0e-9)).all() and t.centroid.shape == (3,)


def test_tetrahedron_volume():
    tetrahedron = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    polynomial_order = 1
    t = Tetrahedron(tetrahedron, polynomial_order)
    expected_volume = 1.0 / 6.0
    assert np.abs(t.volume - expected_volume) < 1.0e-9


polynomial_orders = [1, 2, 3, 4, 5]

functions_tetraherdon = [
    (lambda z, y, x: 2.0 * x ** 1 + 3.0 * y ** 1 + 7.0 * z ** 1 + 3.0),
    (lambda z, y, x: 1.0 * x ** 2 + 6.0 * x + 3.0 * y ** 2 - 2.0 * y + 2.0 * x * y - 2.0 * y * z + 1.0 + 3.0 * z ** 2),
    (lambda z, y, x: 4.0 * x ** 3 - 4.0 * x ** 2 + 5.0 * x - 1.0 + 4.0 * y ** 3),
    (lambda z, y, x: 7.0 * x ** 4 - 4.0 * x ** 3 + 5.0 * x ** 2 - 1.0 * x + 9.0 + 5.0 * y ** 4),
    (
        lambda z, y, x: 2.0 * x ** 5
        - 1.0 * x ** 4
        + 1.0 * x ** 3
        - 3.0 * x ** 2
        + 1.0 * x
        - 6.0
        + 6.0 * y ** 5
        + z ** 5
        + y * z * x
    ),
]

test_data = []
for k in polynomial_orders:
    test_data.append(
        (
            k,
            integrate.tplquad(
                functions_tetraherdon[k - 1],
                0.0,
                1.0,
                lambda x: 0.0,
                lambda x: 1.0 - x,
                lambda x, y: 0.0,
                lambda x, y: 1.0 - x - y,
            )[0],
        )
    )


@pytest.mark.parametrize("k, expected", test_data)
def test_tetrahedron_quadrature(k, expected):
    tetrahedron = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    t = Tetrahedron(tetrahedron, k)
    numerical_integral = np.sum(
        [
            quadrature_weight
            * functions_tetraherdon[k - 1](quadrature_point[2], quadrature_point[1], quadrature_point[0])
            for quadrature_point, quadrature_weight in zip(t.quadrature_nodes, t.quadrature_weights)
        ]
    )
    assert np.abs(numerical_integral - expected) < 1.0e-9
