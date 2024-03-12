"""Plot function for the QuadWire model"""

#%% import packages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%% plot function 

def plotMesh(axis, L, x, elems, edgecolor='black', color='white', clrfun=[], outer=True):
    """
    Mesh display function using parallelepiped elements with triangular facets. 
    The colormap can either display quantities expressed at the particules or expressed at the elements

    Parameters
    ----------
    axis : matplotlib axes 3D 
        axis of the matplotlib 3D plot.
    L : int 
        Length of the wire in mm.
    x : array of size (nNodesTOT, nCoord, nParticules)
        Coordinates of each particule of each node of the mesh.
    elems : array of size ((nNodes - 1)*nLayers, elemOrder +1 ) or (nElemTot, elemOrder +1)
        Index of the pair of nodes forming an element.
    edgecolor : str, optional
        Color of the edges of the triangles. The default is 'black'.
    color : str, optional
        Color of the edges of the triangles. The default is 'white'.
    clrfun : array, optional
        Array giving the value that will be used for the colormap. 
        It can either be expressed at each element or at each particule.
        The default is [].
    outer : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    srf : matplotlib 3D polyline
        Plotted surface.
    tri : array of shape (nElemsTot*12, 3)
        Colormap value at each node for each triangle (12 per element) 

    """
    x = np.moveaxis(x, (0, 1, 2), (0, 2, 1)).reshape(4 * x.shape[0], 3)
    # ax.scatter(x[:,0,:].flatten(), x[:,1,:].flatten(), x[:,2,:].flatten())
    # faces are quadrilaterals with outgoing normals (corners defined in the trigo direction)
    lf = 4 * elems[:, [0, 1, 1, 0]] + [2, 2, 0, 0]  # LEFT faces
    rf = 4 * elems[:, [1, 0, 0, 1]] + [3, 3, 1, 1]  # RIGHT faces
    tf = 4 * elems[:, [0, 1, 1, 0]] + [0, 0, 1, 1]  # TOP faces
    bf = 4 * elems[:, [1, 0, 0, 1]] + [3, 3, 2, 2]  # BOTTOM faces
    ff = 4 * elems[:, [1, 1, 1, 1]] + [2, 3, 1, 0]  # FRONT faces
    bkf = 4 * elems[:, [0, 0, 0, 0]] + [3, 2, 0, 1]  # BACK faces
    faces = np.vstack((lf, rf , tf, bf, ff, bkf))  #  )) #
    
    if outer:  # show only outer faces
        tol = L * 1e-6
        ux, idx = np.unique(np.round(x / tol), axis=0, return_inverse=True)
        ufaces = idx[faces]
        f2n = sp.sparse.csr_matrix((np.ones(faces.size), (
        ufaces.flatten(), np.repeat(np.arange(faces.shape[0]), 4, axis=0))))  # node-face connectivity
        f2f = f2n.T @ f2n  # face-face connectivity
        isinner = ((f2f - 4 * sp.sparse.eye(faces.shape[0])) == 4).sum(axis=1).astype(bool)
        faces = faces[np.invert(np.array(isinner).flatten()), :]

    # conversion to a triangulation (matplotlib limitation)
    
    tri = np.vstack((faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]))
    srf = axis.plot_trisurf(x[:, 0].flatten(), x[:, 1].flatten(), x[:, 2].flatten()
                          , triangles=tri
                          , edgecolor=edgecolor
                          , color=color
                          , cmap = 'Spectral_r'
                          , linewidth=0.1
                          , antialiased=True
                          )  # , alpha = .5         , vmax = max(clrfun)          , cmap="OrRd"

    # colored faces based on a color function
    if np.array(clrfun).size != 0: 
        if np.array(clrfun).size == x.shape[0] :
            # print(np.mean(clrfun[tri], axis=1).shape)
            srf.set_array(np.mean(clrfun[tri], axis=1))
        else :
            if outer :
                clrfun = np.tile(clrfun, 6*2)[np.invert(np.array(isinner).flatten()), :]
                # print("trial shape " +str(trial.shape))
                srf.set_array( clrfun )
            else : 
                clrfun = np.tile(clrfun, 6*2)
                # print("trial shape " +str(trial.shape))
                srf.set_array( clrfun)
                
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    
    return srf, tri


def plotpoints(X):
    """
    Displays the material points of a geometry in a matplotlib 3D plot

    Parameters
    ----------
    X : array of size (nNodes, nCoord)
        Coordinates of each nodes in the global reference (x1, x2, x3).

    Returns
    -------
    None.

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # Tracer les points en 3D
    ax.scatter(X[:,0], X[:,1], X[:,2], c='b', marker='o')