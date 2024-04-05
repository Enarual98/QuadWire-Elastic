"""
Script used to compare the free energy per unit length computed with the QuadWire model 
and comparison for each layer between the QuadWire and the 3D model.

The comparison in the article is made on the sinusoidal structure with a linear thermal loading.
"""
#%%
# clear workspace
from IPython import get_ipython
get_ipython().magic('reset -sf')

#%% Imports
import time
import numpy as np
import scipy as sp
from scipy import sparse
from scipy import spatial
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import qw_structure as qw
from modules import mesh, fem, weld, behavior, plot, solver3D

#%% Parameters
### Geometry
L = 5
Hn = 0.1
Hb = 0.1
buildDirection = "v"

### Mesh
nLayers = 5
nNodes = 301
nNodes3D = 51

### Integration
elemOrder = 1
quadOrder = elemOrder + 1 

### Material behavior
E = 3e3  # Young modulus /!\ in MPa !n
nu = 0.3  # poisson ratio
alpha = 1.13e-5  # thermal expansion coefficient
behavior_opt = True

### Thermal variation
dT = -60

### Plot
wePlot = True
etiquettes = ["     Bead " + str(k) for k in np.arange(1, nLayers +1 )]
etiquettes.append('')
major_ticks = np.arange(0, nLayers*(nNodes), nNodes-1)
color1 = '#3498db'
color2 = '#e74c3c'


#%% Case 1 : Linear wall and uniform thermal loading 
### Studied case
meshType = "linear" # linear wall
thermalLoading = "uniform" # uniform thermal loading
clrmap = "nrg" #  "stt" # nrg for energy and stt for tension

### Computation
## QuadWire
U_rectUniform_opt, Eps_rectUniform_opt, Sigma_rectUniform_opt, e_rectUniform_opt = qw.structure(L, Hn, Hb, meshType, layerType, zig, nNodes, nLayers_h, buildDirection, dT, thermalLoading, elemOrder, quadOrder, behavior_opt, wePlot, clrmap,
                                                                                                5)
## QuadWire
U_rectUniform_hom, Eps_rectUniform_hom, Sigma_rectUniform_hom, e_rectUniform_hom = qw.structure(L, Hn, Hb, meshType, layerType, zig, nNodes, nLayers_h, buildDirection, dT, thermalLoading, elemOrder, quadOrder, behavior_opt, wePlot, clrmap,
                                                                                                5)
## 3D
U3D_rectUniform, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D = solver3D.cauchyTM(L, Hn, Hb, nNodes3D, nLayers, meshType, thermalLoading)
e3D_rectUniform = solver3D.psi_3dtm(U3D_rectUniform, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D, alpha, Hn, Hb, nNodes3D, nLayers, False)

### Energy plot
plt.figure("Energy case 1 : linear wall - uniform thermal loading")
plt.plot(e_rectUniform_opt, color = 'k', label ="QuadWire opt")
plt.plot(e_rectUniform_hom, color = 'grey', label ="QuadWire hom")
plt.plot(np.arange((nNodes3D-1)*nLayers)*6,  e3D_rectUniform, "--r", label ="3D model")
plt.xticks(major_ticks, labels=etiquettes, rotation=45, ha='left', fontsize=12)
plt.ylabel("$\psi$ [J.mm$^{-1}$]", fontsize=13)
plt.grid()
plt.legend()
plt.title("linear wall - uniform thermal loading")

#%% Case 2 : Linear wall and linear thermal loading 
### Studied case
meshType = "linear"
thermalLoading = "linear"
clrmap = "nrg" # "stt" # nrg for energy and stt for tension

### Computation
## QuadWire
U_rectLinear_opt, Eps_rectLinear_opt, Sigma_rectLinear_opt, e_rectLinear_opt = qw.structure(L, Hn, Hb, meshType, layerType, zig, nNodes, nLayers_h, buildDirection, dT, thermalLoading, elemOrder, quadOrder, behavior_opt, wePlot, clrmap, 10)
## QuadWire
U_rectLinear_hom, Eps_rectLinear_hom, Sigma_rectLinear_hom, e_rectLinear_hom = qw.structure(L, Hn, Hb, meshType, layerType, zig, nNodes, nLayers_h, buildDirection, dT, thermalLoading, elemOrder, quadOrder, False, wePlot, clrmap, 10)
## 3D
U3D_rectLinear, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D = solver3D.cauchyTM(L, Hn, Hb, nNodes3D, nLayers, meshType, thermalLoading)
e3D_rectLinear = solver3D.psi_3dtm(U3D_rectLinear, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D, alpha, Hn, Hb, nNodes3D, nLayers, False)

### Plot
plt.figure("Energy case 2 : linear wall - linear thermal loading")
plt.plot(e_rectLinear_opt, color = 'k', label ="QuadWire opt")
plt.plot(e_rectLinear_hom, color = 'grey', label ="QuadWire hom")
plt.plot(np.arange(0, nLayers*(nNodes3D-1))*6, e3D_rectLinear, "--r", label ="3D model")
plt.xticks(major_ticks, labels=etiquettes, rotation=45, ha='left', fontsize=12)
plt.ylabel("$\psi$ [J.mm$^{-1}$]", fontsize=14)
plt.grid()
plt.legend()
plt.title("linear wall - linear thermal loading")

#%% Case study 3 : Sinusoidal wall and uniform thermal loading
### Studied case
meshType = "sinus"
thermalLoading = "uniform"
clrmap = "nrg" #  "stt" # nrg for energy and stt for tension

### Computation
## QuadWire (optimized behavior)
U_sinUniform_opt, Eps_sinUniform_opt, Sigma_sinUniform_opt, e_sinUniform_opt = qw.structure(L, Hn, Hb, meshType, layerType, zig, nNodes, nLayers_h, buildDirection, dT, thermalLoading, elemOrder, quadOrder, behavior_opt, wePlot, clrmap, 2)
## QuadWire (homogenized behavior)
U_sinUniform_hom, Eps_sinUniform_hom, Sigma_sinUniform_hom, e_sinUniform_hom = qw.structure(L, Hn, Hb, meshType, layerType, zig, nNodes, nLayers_h, buildDirection, dT, thermalLoading, elemOrder, quadOrder, False, wePlot, clrmap, 2)
## 3D
U3D_sinUniform, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D = solver3D.cauchyTM(L, Hn, Hb, nNodes3D, nLayers, meshType, thermalLoading)
e3_sinUniform = solver3D.psi_3dtm(U3D_sinUniform, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D, alpha, Hn, Hb, nNodes3D, nLayers, False)

### Plot
plt.figure("Energy case 3 : sinus wall - uniform thermal loading")
plt.plot(e_sinUniform_opt, color = 'k', label ="QuadWire opt")
plt.plot(e_sinUniform_hom, color = 'grey', label ="QuadWire hom")
plt.plot(np.arange(0, nLayers*(nNodes3D-1))*6, e3_sinUniform, "--r", label ="3D model")
plt.xticks(major_ticks, labels=etiquettes, rotation=45, ha='left', fontsize=12)
plt.ylabel("$\psi$ [J.mm$^{-1}$]", fontsize=14)
plt.grid()
plt.legend()
plt.title("sinus wall - uniform thermal loading")


#%% Case study 4 : Sinusoidal wall and linear thermal loading
### Studied case
meshType = "sinus"
thermalLoading = "linear"
clrmap = "nrg" #  "stt" # nrg for energy and stt for tension

### Computation
## QuadWire (optimized behavior)
tic = time.time()
U_sinLinear_opt, Eps_sinLinear_opt, Sigma_sinLinear_opt, e_sinLinear_opt = qw.structure(L, Hn, Hb, meshType, layerType, zig, nNodes, nLayers_h, buildDirection, dT, thermalLoading, elemOrder, quadOrder, behavior_opt, wePlot, clrmap, 2)
tac = time.time()
print('QW :', tac-tic)
## QuadWire (homogenized behavior)
tic = time.time()
U_sinLinear_hom, Eps_sinLinear_hom, Sigma_sinLinear_hom, e_sinLinear_hom = qw.structure(L, Hn, Hb, meshType, layerType, zig, nNodes, nLayers_h, buildDirection, dT, thermalLoading, elemOrder, quadOrder, False, wePlot, clrmap, 2)
tac = time.time()
print('QW :', tac-tic)
## 3D
tic = time.time()
U3D_sinLinear, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D = solver3D.cauchyTM(L, Hn, Hb, nNodes3D, nLayers, meshType, thermalLoading)
e3D_sinLinear = solver3D.psi_3dtm(U3D_sinLinear, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D, alpha, Hn, Hb, nNodes3D, nLayers, False)
tac = time.time()
print('3D :', tac-tic)

### Plot
plt.figure("Energy case 4 : sinus wall - linear thermal loading")
plt.plot(e_sinLinear_opt, color = 'k', label ="QuadWire opt")
plt.plot(e_sinLinear_hom, color = 'grey', label ="QuadWire hom")
plt.plot(np.arange(0, nLayers*(nNodes3D-1))*6, e3D_sinLinear, "--r", label ="3D model")
plt.xticks(major_ticks, labels=etiquettes, rotation=0, ha='left', fontsize=12)
plt.ylabel("$\psi$ [J.mm$^{-1}$]", fontsize=14)
plt.grid()
plt.legend()
plt.title("sinus wall - linear thermal loading")


 