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
import numpy as np
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
U_rectUniform, Eps_rectUniform, Sigma_rectUniform, e_rectUniform = qw.structure(L, Hn, Hb, meshType, nNodes, nLayers, buildDirection, dT, thermalLoading, elemOrder, quadOrder, behavior_opt, wePlot, clrmap, 5)
## 3D
U3D_rectUniform, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D = solver3D.cauchyTM(L, Hn, Hb, nNodes3D, nLayers, meshType, thermalLoading)
e3D_rectUniform = solver3D.psi_3dtm(U3D_rectUniform, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D, alpha, Hn, Hb, nNodes3D, nLayers, False)

### Energy plot
plt.figure("Energy case 1 : linear wall - uniform thermal loading")
plt.plot(e_rectUniform, color=color1, label ="QW")
plt.plot(np.arange((nNodes3D-1)*nLayers)*6,  e3D_rectUniform, color=color2, label ="3DTM")
plt.xticks(major_ticks, labels=etiquettes, rotation=45, ha='left', fontsize=12)
plt.ylabel("$\psi$ [J.mm$^{-1}$]", fontsize=13)
plt.grid()
plt.legend()

#%% Case 2 : Linear wall and linear thermal loading 
### Studied case
meshType = "linear"
thermalLoading = "linear"
clrmap = "nrg" # "stt" # nrg for energy and stt for tension

### Computation
## QuadWire
U_rectLinear, Eps_rectLinear, Sigma_rectLinear, e_rectLinear = qw.structure(L, Hn, Hb, meshType, nNodes, nLayers, buildDirection, dT, thermalLoading, elemOrder, quadOrder, behavior_opt, wePlot, clrmap, 10)
## 3D
U3D_rectLinear, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D = solver3D.cauchyTM(L, Hn, Hb, nNodes3D, nLayers, meshType, thermalLoading)
e3D_rectLinear = solver3D.psi_3dtm(U3D_rectLinear, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D, alpha, Hn, Hb, nNodes3D, nLayers, False)

### Plot
plt.figure("Energy case 2 : linear wall - linear thermal loading")
plt.plot(e_rectLinear, color = color1, label ="QW ")
plt.plot(np.arange(0, nLayers*(nNodes3D-1))*6, e3D_rectLinear, color = color2, label ="3DTM")
plt.xticks(major_ticks, labels=etiquettes, rotation=45, ha='left', fontsize=12)
plt.ylabel("$\psi$ [J.mm$^{-1}$]", fontsize=14)
plt.grid()
plt.legend()

#%% Case study 3 : Sinusoidal wall and uniform thermal loading
### Studied case
meshType = "sinus"
thermalLoading = "uniform"
clrmap = "nrg" #  "stt" # nrg for energy and stt for tension

### Computation
## QuadWire
U_sinUniform, Eps_sinUniform, Sigma_sinUniform, e_sinUniform = qw.structure(L, Hn, Hb, meshType, nNodes, nLayers, buildDirection, dT, thermalLoading, elemOrder, quadOrder, behavior_opt, wePlot, clrmap, 2)
## 3D
U3D_sinUniform, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D = solver3D.cauchyTM(L, Hn, Hb, nNodes3D, nLayers, meshType, thermalLoading)
e3_sinUniform = solver3D.psi_3dtm(U3D_sinUniform, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D, alpha, Hn, Hb, nNodes3D, nLayers, False)

### Plot
plt.figure("Energy case 2 : sinus wall - uniform thermal loading")
plt.plot(e_sinUniform, color = color1, label ="QW")
plt.plot(np.arange(0, nLayers*(nNodes3D-1))*6, e3_sinUniform, color = color2, label ="3DTM")
plt.xticks(major_ticks, labels=etiquettes, rotation=45, ha='left', fontsize=12)
plt.ylabel("$\psi$ [J.mm$^{-1}$]", fontsize=14)
plt.grid()
plt.legend()


#%% Case study 4 : Sinusoidal wall and linear thermal loading
### Studied case
meshType = "sinus"
thermalLoading = "linear"
clrmap = "nrg" #  "stt" # nrg for energy and stt for tension

### Computation
## QuadWire
U_sinLinear, Eps_sinLinear, Sigma_sinLinear, e_sinLinear = qw.structure(L, Hn, Hb, meshType, nNodes, nLayers, buildDirection, dT, thermalLoading, elemOrder, quadOrder, behavior_opt, wePlot, clrmap, 2)
## 3D
U3D_sinLinear, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D = solver3D.cauchyTM(L, Hn, Hb, nNodes3D, nLayers, meshType, thermalLoading)
e3D_sinLinear = solver3D.psi_3dtm(U3D_sinLinear, B3D, Y3D, C3D, nQP3D, dTmoy3D, xiQ3D, elemQ3D, alpha, Hn, Hb, nNodes3D, nLayers, False)

### Plot
plt.figure("Energy case 4 : sinus wall - linear thermal loading")
plt.plot(e_sinLinear, color = 'k', label ="QuadWire")
plt.plot(np.arange(0, nLayers*(nNodes3D-1))*6, e3D_sinLinear, "--r", label ="3D model")
plt.xticks(major_ticks, labels=etiquettes, rotation=0, ha='left', fontsize=12)
plt.ylabel("$\psi$ [J.mm$^{-1}$]", fontsize=14)
plt.grid()
plt.legend()


 