from tests import context
import pytest
import numpy as np
from shapes.point import Point


def test_point_centroid():
    point = np.array([[0.2]])
    p = Point(point)
    # expected_centroid = np.array([0.2])
    expected_centroid = np.array([])
    assert (p.centroid == expected_centroid).all() and p.centroid.shape == (0,)


def test_point_volume():
    point = np.array([[0.2]])
    p = Point(point)
    expected_volume = 1.0
    assert p.volume == expected_volume


def test_point_quadrature():
    point = np.array([[0.2]])
    p = Point(point)
    expected_quadrature_nodes = np.array([[]])
    expected_quadrature_weights = np.array([[1.0]])
    assert (
        (p.quadrature_points == expected_quadrature_nodes).all()
        and p.quadrature_points.shape == (1, 0)
        and (p.quadrature_weights == expected_quadrature_weights).all()
        and p.quadrature_weights.shape == (1, 1)
    )
