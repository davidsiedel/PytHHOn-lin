import mgis
import mgis.behaviour as mgis_bv
import os
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# quadrature points
# ----------------------------------------------------------------------------------------------------------------------
quadratures_points_in_mesh = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
number_of_quadratures_points_in_mesh = len(quadratures_points_in_mesh)
# ----------------------------------------------------------------------------------------------------------------------
# mgis
# ----------------------------------------------------------------------------------------------------------------------
lib = "/Users/davidsiedel/Projects/PytHHOn3D/bhv/src/libBehaviour.dylib"
h = mgis_bv.Hypothesis.Tridimensional
b = mgis_bv.load(lib, "Elasticity", h)
# material data manager
m = mgis_bv.MaterialDataManager(b, number_of_quadrature_points_in_mesh)
# 
m.s0.setMaterialProperty("YoungModulus", 1.124999981250001)
m.s0.setMaterialProperty("PoissonRatio", 0.499999999999999)
m.s1.setMaterialProperty("YoungModulus", 1.124999981250001)
m.s1.setMaterialProperty("PoissonRatio", 0.499999999999999)
T = 293.15 * np.ones(number_of_quadrature_points_in_mesh)
Ts = mgis_bv.MaterialStateManagerStorageMode.ExternalStorage
mgis_bv.setExternalStateVariable(m.s0, "Temperature", T, Ts)
mgis_bv.setExternalStateVariable(m.s1, "Temperature", T, Ts)
it = mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator
dt = 1.0 / 16.0
mgis_bv.integrate(m, it, dt, 0, m.n)
mgis_bv.update(m)
for 
for i in range(0, 1):
    it = mgis_bv.IntegrationType.IntegrationWithTangentOperator
    dt = 1.0 / 16.0
    mgis_bv.integrate(m, it, dt, 0, m.n)
    mgis_bv.update(m)
    for p in range(0, number_of_quadrature_points_in_mesh):
        m.s1.gradients[p][0] += de
    m.k -- tangent

