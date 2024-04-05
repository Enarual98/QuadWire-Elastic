"""
Testing welding wires vertically with offset between successive beads
"""
# %%
# clear workspace

from IPython import get_ipython

get_ipython().magic('reset -sf')

# %% Imports
import time
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import qw_additive as qwa
import qw_structure as qws
from modules import mesh, fem, weld, behavior, plot, thermaldata, forces
import generate_thermal

# %% Test ZigZag Parameters

### Integration
elemOrder = 1
quadOrder = elemOrder + 1

### Material behavior
E = 3e3  # Young modulus /!\ in MPa !n
nu = 0.3  # poisson ratio
alpha = 1.13e-5  # thermal expansion coefficient
behavior_opt = True  # use the optimized behavior

### Geometry
L = 5  # length in mm
Hn = 0.1  # width of the section in mm
Hb = 0.1  # height of the section in mm
beadType = "linear"  # linear, circular, square, quarter_circular, quarter_square, sinus
layerType = "normal" # default is "normal" (required for closed geometries : circular, square, quarter_square, quarter_circular (>180°)), also available is "duplicate" (possible for linear and sinus)
zigzag = False

### Mesh Parameters
nLayers_h = 1 # number of horizontal layers (beads)
nLayers_v = 15  # number of vertical layers
nLayers = nLayers_h * nLayers_v  # number of layers

nNodes = 51  # number of nodes per layers
nElems = nNodes-1
lc = L / nElems   #length of elements np.sqrt(12)*lt

offset = mesh.stacking_offset(L, Hn, nNodes, nLayers_v, "exp-sinus", 0, 0.75)  # offset between successive layers along t and n (nLayers_v, 2)

# %%## thermal data
dT = -60
loadType = "linear"
# path = r".\thermal_data\wall"     #path = r".\thermal_data\fake_data"   #
# generate_thermal.generate_fake_data(nElems*nLayers, path)


# %%
X, Elems, U0 = mesh.mesh_first_bead(L, nNodes, beadType)
#plot.plotpoints(X)
# %%
X, Elems, U0 = mesh.mesh_first_layer(X, Elems, U0, nNodes, nLayers_h, nLayers_v, Hn, Hb, layerType, zigzag)
#plot.plotpoints(X)
# %%
X, Elems, U0 = mesh.mesh_structure(X, Elems, U0, nNodes, nLayers_h, nLayers_v, Hn, Hb, zigzag)
plot.plotpoints(X)
# %%
X = mesh.mesh_offset(X, Elems, nNodes, nLayers_h, nLayers_v, zigzag, offset, layerType)
plot.plotpoints(X)


# %% Computation
scalefig = 1

tic = time.time()
# U, Eps, Sigma, elementEnergyDensity, qp2elem, nQP, x, Elems = qwa.additive(path, L, Hn, Hb, beadType, layerType, offset, zigzag, nNodes, nLayers_h, nLayers_v, elemOrder, quadOrder, behavior_opt, toPlot=True, clrmap="temp", scfplot = scalefig)
U, Eps, Sigma, elementEnergyDensity, qp2elem, nQP, x, Elems = qws.structure(L, Hn, Hb, beadType, layerType, offset, zigzag, nNodes, nLayers_h, nLayers_v, dT, loadType, elemOrder, quadOrder, behavior_opt, toPlot=False, clrmap="stt", scfplot=scalefig)
tac = time.time()
print('computing time :', tac - tic, 's, ', (tac - tic) / 60, 'min')



# %% Plot stt
projection_index = 0
sigmaplot = qp2elem @ Sigma[projection_index*nQP:(projection_index+1)*nQP]
clr = sigmaplot[:,0]
clr = clr / (Hn * Hb)


fig = plt.figure()
ax = plt.axes(projection='3d', proj_type='ortho')
ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1]*scalefig), np.ptp(x[:, 2]*scalefig)))

srf, tri = plot.plotMesh(ax, L, x, Elems, color='none', edgecolor='black', clrfun=clr, outer=False)
colorbar = plt.colorbar(srf, pad=0.15)
colorbar.set_label('$\sigma_{tt}$ [MPa]')

plt.show()