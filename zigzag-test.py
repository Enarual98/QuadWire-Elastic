"""
Testing ZigZag printing strategy with the QuadWire model.
Thermal data is wrong (wall or carpet but not zigzag)
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
from modules import mesh, fem, weld, behavior, plot, thermaldata, forces

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
L = 10  # length in mm
Hn = 0.1  # width of the section in mm
Hb = 0.1  # height of the section in mm
meshType = "linear"  # linear for a rectilinear wall - other options are sinus, circular
zigzag = False

### Mesh Parameters
path = r".\thermal_data\wall"
nLayers_h = 3  # number of horizontal layers (beads)
nLayers_v = 5  # number of vertical layers
nLayers = nLayers_h * nLayers_v  # number of layers
nNodes = 51  # number of nodes per layers
offset = mesh.stacking_offset(L, Hn, nNodes, nLayers_v, "linear", 0, 1)  # offset between successive layers along t and n (nLayers_v, 2)

### Plot
toPlot = True  # True if you want to see the evolution of the field during printing
colormap = "temp"  # Field to observe on the graph. Options are stt stn stb temp

scalefig = 10

color1 = "#00688B"
color2 = "#B22222"
color3 = "#556B2F"

# %% ZigZag test

### Computation
tic = time.time()
U, Eps, Sigma, elementEnergyDensity, qp2elem, nQP, x, Elems = qwa.additive(path, L, Hn, Hb, meshType, layerType, offset, zigzag, nNodes, nLayers_h, nLayers_v, elemOrder, quadOrder, behavior_opt, toPlot, clrmap=colormap, scfplot=scalefig)
tac = time.time()
print('computing time :', tac - tic, 's, ', (tac - tic) / 60, 'min')

# %% Plot
### Plot stt
projection_index = 0
sigmaplot = qp2elem @ Sigma[projection_index * nQP:(projection_index + 1) * nQP]
clr = sigmaplot[:, 0]
clr = clr / (Hn * Hb)

fig = plt.figure()
ax = plt.axes(projection='3d', proj_type='ortho')
ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1] * scalefig), np.ptp(x[:, 2] * scalefig)))
ax.set_title(zigzag * 'ZigZag' + (1 - zigzag) * 'ZigZig')

srf, tri = plot.plotMesh(ax, L, x, Elems, color='none', edgecolor='none', clrfun=clr, outer=False)
colorbar = plt.colorbar(srf, pad=0.15)
colorbar.set_label('$\sigma_{tt}$ [MPa]')

### Plot stn
projection_index = 3
sigmaplot = qp2elem @ Sigma[projection_index * nQP:(projection_index + 1) * nQP]
clr = sigmaplot[:, 0]
clr = clr / (Hn * Hb)

fig = plt.figure()
ax = plt.axes(projection='3d', proj_type='ortho')
ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1] * scalefig), np.ptp(x[:, 2] * scalefig)))
ax.set_title(zigzag * 'ZigZag' + (1 - zigzag) * 'ZigZig')

srf, tri = plot.plotMesh(ax, L, x, Elems, color='none', edgecolor='none', clrfun=clr, outer=False)
colorbar = plt.colorbar(srf, pad=0.15)
colorbar.set_label('$\sigma_{tn}$ [MPa]')

### Plot stb
projection_index = 4
sigmaplot = qp2elem @ Sigma[projection_index * nQP:(projection_index + 1) * nQP]
clr = sigmaplot[:, 0]
clr = clr / (Hn * Hb)

fig = plt.figure()
ax = plt.axes(projection='3d', proj_type='ortho')
ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1] * scalefig), np.ptp(x[:, 2] * scalefig)))
ax.set_title(zigzag * 'ZigZag' + (1 - zigzag) * 'ZigZig')

srf, tri = plot.plotMesh(ax, L, x, Elems, color='none', edgecolor='none', clrfun=clr, outer=False)
colorbar = plt.colorbar(srf, pad=0.15)
colorbar.set_label('$\sigma_{tb}$ [MPa]')  #
# sigtb = Sigma[projection_index*nQP:(projection_index+1)*nQP]/(Hn*Hb)
#
# x_list = np.linspace(-L/2, L/2, int(nQP/nLayers))
# plt.figure()
# plt.ylabel("$\sigma_{tb}$ [MPa]")
# plt.xlabel("s [mm]")
# plt.plot(x_list, sigtb[:int(nQP/nLayers)], color=color1, label = "First layer")
# plt.plot(x_list, sigtb[int(nQP/nLayers)*(nLayers-8):int(nQP/nLayers)*(nLayers-7)], color=color2, label = "Middle layer")
# plt.plot(x_list, sigtb[int(nQP/nLayers)*(nLayers-1):], color=color3, label = "Last layer")
# plt.xticks(np.linspace(-L/2, L/2, L//10))
# plt.legend()
# plt.grid()
