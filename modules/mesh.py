""" Module containing the discretisation and meshing functions of the QuadWire program """

# %% import packages

import numpy as np
import scipy as sp


# %% Discretization and meshing
def local_axis(X, Elems):
    """
    Derive local axis t (tangent to bead trajectory) and n (normal to bead trajectory) at each node
    Parameters
    ----------
    X : array of size (nNodes, nCoord)
        Coordinates of each node in the global reference (x1, x2, x3).
    Elems : array of size (nNodes - 1, 2)
        Index of the pair of nodes forming an element.

    Returns
    -------
    t : array of size (nNodes, nCoord)
        tangent vector
    n : array of size (nNodes, nCoord)
        transverse vector
    """
    t = np.diff(X[Elems, :], 1, axis=1).reshape((Elems.shape[0], X.shape[1]))  # element tangent vector (nElems, nCoord)
    t = t / np.sqrt(np.sum(t ** 2, 1))[:, np.newaxis]  # normalization
    e2n = sp.sparse.csr_matrix((np.ones((Elems.size,)), (Elems.flatten(), np.repeat(np.arange(Elems.shape[0]), Elems.shape[1], 0))))  # elem-node connectivity matrix (nNodes, nElems)
    t = e2n @ t  # (nNodes, nElems)
    t = t / np.sqrt(np.sum(t ** 2, 1))[:, np.newaxis]  # re-normalization
    n = t @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])  # normal vector rotated of 90° in the XY plane
    b = np.cross(t, n)
    return t, n


def mesh_first_bead(L, nNodes, beadType="linear", meshRefined=False):
    """
    Function that meshes the first bead of the structure.

    Parameters
    ----------
    L : int
        Length of the structure in tangential direction t.
    nNodes : int
        Number of nodes discretising the length L.
    beadType : str, optional
        Geometry of the first bead. Options are "linear", "circular", "square", "quarter_circular", "quarter_square", "sinus"
        The default is "linear". "circular" and "square" are closed shapes.
    meshRefined : bool, optional
        Option available for a "linear" and "sinus" mesh : refinement of the mesh close to the edges of the structure.
        The default is False.

    Returns
    -------
    X : array of size (nNodes, nCoord)
        Coordinates of each node in the global reference (x1, x2, x3).
    Elems : array of size (nNodes - 1, 2) for open beadType or (nNodes, 2) for closed beadType
        Index of the pair of nodes forming an element.
    U0 : array of size (nCoord, nParticule, nNodes)
        Boundary conditions for particles attached to each node : NaN when the particule has no imposed bc. 
        Particules are indexed as {TL,TR,BL,BR}. 

    """
    if meshRefined:
        p = 2
        x1 = np.tanh(np.linspace(-p, p, nNodes)) / np.tanh(p) * L / 2  # heterogeneous mesh
    else:
        x1 = np.linspace(-1, 1, nNodes) * L / 2  # uniform mesh

    if beadType == "linear":
        X = x1[:, np.newaxis] * [1, 0, 0]  # straight line along X1

    elif beadType == "circular":
        theta = np.arange(nNodes)[:, np.newaxis] / nNodes * 2 * np.pi
        R = L / 2 / np.pi
        X = np.hstack((R * np.cos(theta), R * np.sin(theta), theta * 0))

    elif beadType == "square":
        if nNodes % 4 != 0:
            raise ValueError("Le nombre de noeuds par couche renseigné n'est pas divisible par 4")
        dl = L / nNodes
        c = nNodes // 4
        X = np.zeros((nNodes, 3))

        # Create arrays for indices
        i = np.repeat(np.arange(4), c)
        j = np.tile(np.arange(c), 4)
        # Calculate x coordinates
        X[:, 0] = j * dl * (i == 0) + c * dl * (i == 1) + (c - j) * dl * (i == 2)
        # Calculate y coordinates
        X[:, 1] = j * dl * (i == 1) + c * dl * (i == 2) + (c - j) * dl * (i == 3)

    elif beadType == "quarter_circular":
        t_val = 250
        theta = np.linspace(np.pi / 2, np.pi / 2 - (2 * np.pi * t_val / 360), nNodes)[:, np.newaxis]
        # theta = np.arange(nNodes)[:,np.newaxis]/nNodes* (2*np.pi*t_val/360)
        R = L / 2 / np.pi
        X = np.hstack((R * np.cos(theta), R * np.sin(theta), theta * 0))  # straight line along X1

    elif beadType == "quarter_square":
        nNodes_mod4_sup = np.ceil(nNodes / 4) * 4
        dl = L / nNodes
        c = nNodes_mod4_sup // 4
        X = np.zeros((nNodes, 3))

        # Create arrays for indices
        i = np.repeat(np.arange(4), c)
        j = np.tile(np.arange(c), 4)
        # Calculate x coordinates
        X[:, 0] = (j * dl * (i == 0) + c * dl * (i == 1) + (c - j) * dl * (i == 2))[:nNodes]
        # Calculate y coordinates
        X[:, 1] = (j * dl * (i == 1) + c * dl * (i == 2) + (c - j) * dl * (i == 3))[:nNodes]

    elif beadType == "sinus":  # sinus shape leads to bead length > L
        amplitude = 0.25 * L
        frequency = 1.0 / L
        omega = 2 * np.pi * frequency
        X = np.concatenate((x1[:, np.newaxis], amplitude * np.sin(x1 * omega)[:, np.newaxis], x1[:, np.newaxis] * 0), axis=1)

    else:
        raise ValueError("La maillage renseigné n'est pas connu")

    ### Define elements
    if beadType == "circular" or beadType == "square":
        Elems = np.mod(np.arange(nNodes)[:, np.newaxis] + [0, 1], nNodes)  # simple elements
    else:
        Elems = np.arange(nNodes - 1)[:, np.newaxis] + [0, 1]  # simple elements

    ### Define boundary conditions on particles (NaN if no boundary condition). Array of size (nCoord, nParticule, nNodes).
    # U0 = np.tile([[[np.nan], [np.nan], [np.nan], [np.nan]]], (3, 1, nNodes)) # glisse
    # U0[:, :, -1] = np.tile([[[0], [0], [0], [0]]], (3, 1)).reshape((3, 4))
    U0 = np.tile([[[np.nan], [np.nan], [0], [0]]], (3, 1, nNodes))  # clamp all {BL,BR} wires

    return X, Elems, U0


def mesh_first_layer(X, Elems, U0, nNodes, nLayers_h, nLayers_v, Hn, Hb, layerType="normal", zigzag=False):
    """
    Function that meshes the first layer of the structure.

    Parameters
    ----------
    X : array of size (nNodes, nCoord)
        Coordinates of each node in the global reference (x1, x2, x3).
    Elems : array of size (nNodes - 1, 2) for open beadType or (nNodes, 2) for closed beadType
        Index of the pair of nodes forming an element.
    U0 : array of size (nCoord, nParticule, nNodes)
        Boundary conditions for particles attached to each node : NaN when the particule has no imposed bc.
        Particules are indexed as {TL,TR,BL,BR}.
    nNodes : int
        Number of nodes discretising the length L of the first bead.
    nLayers_h : int
        Number of layers of the structure in the horizontal direction.
    nLayers_v : int
        Number of layers of the structure in the build direction.
    Hn : float
        Width of the section of the bead.
    Hb : float
        Height of the section of the bead.
    layerType : "normal" or "duplicate"
        default is "normal" (required for closed geometries : circular, square, quarter_square, quarter_circular (>180°)), also available is "duplicate" (possible for linear and sinus)
        beads are either duplicated and translated next to eachother ("duplicate") or stretched/shrinked and translated along the local normal vector ("normal").
    zigzag : bool
        If True then successive beads are printed in back and forth directions.

    Returns
    -------
    X : array of size (nNodes, nCoord)
        Coordinates of each node in the global reference (x1, x2, x3).
    Elems : array of size (nNodes - 1, 2) for open beadType or (nNodes, 2) for closed beadType
        Index of the pair of nodes forming an element.
    U0 : array of size (nCoord, nParticule, nNodes)
        Boundary conditions for particles attached to each node : NaN when the particule has no imposed bc.
        Particules are indexed as {TL,TR,BL,BR}.

    """
    if layerType == "duplicate":  # duplicate bead and translate with Hn offset
        if not zigzag:
            X = np.tile(X, (nLayers_h, 1))
        if zigzag and nLayers_h > 1:  # tile alternating forward (X) and backward (X[::-1, :]) beads
            X = np.tile(np.concatenate((X, X[::-1, :]), axis=0), (nLayers_h // 2, 1))
            if nLayers_h % 2:  # if uneven layer number, add one last forward bead
                X = np.concatenate((X, X[:nNodes, :]), axis=0)
        X += np.repeat(np.arange(nLayers_h)[:, np.newaxis] * [0, Hn, 0], nNodes, axis=0)

    elif layerType == "normal":  # duplicate bead and translate with Hn offset along local normal vector (leads to stretching / shrinking of beads)
        t, n = local_axis(X, Elems)
        X = np.tile(X, (nLayers_h, 1)) + np.repeat(Hn * np.arange(nLayers_h)[:, np.newaxis], nNodes, axis=0) * np.tile(n, (nLayers_h, 1))
        if zigzag:
            Xreshape = X.reshape(nLayers_h, nNodes, 3)
            Xzig = Xreshape[::2]
            Xzag = Xreshape[1::2]
            Xzag = Xzag[:, ::-1, :]
            Xreshape[1::2] = Xzag
            X = Xreshape.reshape(nLayers_h * nNodes, 3)

    ## Replicate the elements
    Elems = np.tile(Elems, (nLayers_h, 1)) + nNodes * np.repeat(np.arange(nLayers_h)[:, np.newaxis], Elems.shape[0], axis=0)
    ## Replicate boundary conditions (NaN when the particule has no imposed bc)
    U0 = np.concatenate((U0, np.tile(U0, (1, 1, nLayers_h - 1))), axis=2)

    return X, Elems, U0


def mesh_structure(X, Elems, U0, nNodes, nLayers_h, nLayers_v, Hn, Hb, zigzag=False):
    """
    Function that replicates the mesh of the first layer to generate the mesh of the whole structure

    Parameters
    ----------
    X : array of size (nNodes, 3)
        Coordinates of each node in the global reference (t,n,b).
    Elems : array of size (nNodes - 1, 2)
        Index of the pair of nodes forming an element.
    U0 : array of size (3, 4, nNodes)
        Boundary conditions for particles attached to each node : NaN when the particule has no imposed bc. 
        Particules are indexed as {TL,TR,BL,BR}.
    nNodes : int
        Number of nodes discretising the length L of the first bead.
    nLayers_h : int
        Number of layers of the structure in the horizontal direction.
    nLayers_v : int
        Number of layers of the structure in the build direction.
    Hn : float
        Width of the section of the bead.
    Hb : float
        Height of the section of the bead.
    zigzag : bool
        If True then successive beads are printed in back and forth directions.

    Returns
    -------
    X : array of size (nNodes*nLayers, 3)
        Coordinates of each node in the global reference (t,n,b).
    Elems : array of size ((nNodes - 1)*nLayers, 2)
        Index of the pair of nodes forming an element.
    U0 : array of size (3, 4, nNodes*nLayers)
        Boundary conditions for particles attached to each node : NaN when the particule has no imposed bc. Particules are indexed as {TL,TR,BL,BR}. 
    """

    ## Replicate the bead in the building direction

    if not zigzag:
        X = np.tile(X, (nLayers_v, 1))
    elif zigzag and nLayers_h == 1:  # tile alternating forward (X) and backward (X[::-1, :]) beads
        X = np.tile(np.concatenate((X, X[::-1, :]), axis=0), (nLayers_v // 2, 1))
        if nLayers_v % 2:  # if uneven layer number, add one last forward bead
            X = np.concatenate((X, X[:nNodes, :]), axis=0)
    else:  # zigzag and nLayers_h > 1
        # initial layer is Xzig_carpet
        Xzig_carpet = X.copy()
        # reverse initial layer to create Xzag_carpet
        Xreshape = X.reshape(nLayers_h, nNodes, 3)
        Xzig, Xzag = Xreshape[::2], Xreshape[1::2]
        Xzig, Xzag = Xzig[:, ::-1, :], Xzag[:, ::-1, :]
        Xreshape[::2] = Xzig
        Xreshape[1::2] = Xzag
        Xzag_carpet = Xreshape.reshape(nLayers_h * nNodes, 3)
        # alternate Xzig_carpet and Xzag_carpet
        X = np.tile(np.concatenate((Xzig_carpet, Xzag_carpet), axis=0), (nLayers_v // 2, 1))
        if nLayers_v % 2:  # if uneven layer number, add one last carpet
            X = np.concatenate((X, Xzig_carpet), axis=0)
    # add Hb offset
    X += np.repeat(np.arange(nLayers_v)[:, np.newaxis] * [0, 0, Hb], nNodes * nLayers_h, axis=0)

    ## Replicate boundary conditions (NaN when the particule has no imposed bc)
    U0 = np.concatenate((U0, np.tile(np.nan * U0, (1, 1, nLayers_v - 1))), axis=2)
    ## Replicate the elements
    Elems = np.tile(Elems, (nLayers_v, 1)) + nLayers_h * nNodes * np.repeat(np.arange(nLayers_v)[:, np.newaxis], Elems.shape[0], axis=0)

    return X, Elems, U0


def welding_conditions(X, Elems, U0, nNodes, nLayers_h, nLayers_v, Hn, Hb, zigzag):
    """
    Function that returns a matrix of wire welding conditions of the structure
    
    Parameters
    ----------
    X : array of size (nNodes, 3)
        Coordinates of each node in the global reference (t,n,b).
    Elems : array of size (nNodes - 1, 2)
        Index of the pair of nodes forming an element.
    U0 : array of size (3, 4, nNodes)
        Boundary conditions for particles attached to each node : NaN when the particule has no imposed bc. 
        Particules are indexed as {TL,TR,BL,BR}. 
    nLayers_h : int
        Number of layers of the structure in the horizontal direction.
    nLayers_v : int
        Number of layers of the structure in the build direction.
    nNodes : int
        Number of nodes discretising the length L.
    Hn : float
        Width of the section of the bead.
    Hb : float
        Height of the section of the bead.
    zigzag : bool
        If True then successive beads are printed in back and forth directions.
        
    Returns
    -------
    weldDOF : array of size (nNodes*(nLayers-1)*2, 4) - number of layer interface * two welding conditions
        Array listing the welding conditions between the wires 
        The k^th welding condition is given by : weldDOF[k] = [indexNode1, node1Particule, indexNode2, node2Particule] which means that the node1Particule chosen is welded to the node2Particule
        There are nInterface*nNodes*2 welding conditions in between wires
    """

    if nLayers_v == 1:  # carpet
        weldNodes = np.arange(nNodes)[:, np.newaxis] * [1, 1] + [0, nNodes]
        if zigzag:
            weldNodes[:, 0] = weldNodes[::-1, 0]
        weldNodes = np.tile(weldNodes, (nLayers_h - 1, 1)) + np.repeat(nNodes * np.arange(nLayers_h - 1)[:, np.newaxis], nNodes, axis=0)
        ## welding TR to TL and BR to BL: weldDOF = [ [[iNode,1,iNode+nNodes,0],[iNode,3,iNode+nNodes,2]] for iNode in range(nNodes)]
        weldDOF = np.tile(weldNodes[:, [0, 0, 1, 1]] * [1, 0, 1, 0], (2, 1)) + np.repeat([[0, 1, 0, 0], [0, 3, 0, 2]], weldNodes.shape[0], axis=0)  # horizontal welding (next to each other)

    elif nLayers_h == 1:  # thin wall
        weldNodes = np.arange(nNodes)[:, np.newaxis] * [1, 1] + [0, nNodes]
        if zigzag:
            weldNodes[:, 0] = weldNodes[::-1, 0]
        weldNodes = np.tile(weldNodes, (nLayers_v - 1, 1)) + np.repeat(nNodes * np.arange(nLayers_v - 1)[:, np.newaxis], nNodes, axis=0)
        ## welding TL to BL and TR to BR: weldDOF = [ [[iNode,0,iNode+nNodes,2],[iNode,1,iNode+nNodes,3]] for iNode in range(nNodes)]
        weldDOF = np.tile(weldNodes[:, [0, 0, 1, 1]] * [1, 0, 1, 0], (2, 1)) + np.repeat([[0, 0, 0, 2], [0, 1, 0, 3]], weldNodes.shape[0], axis=0)  # vertical welding (on top of each other)

    else:  # any nLayers_h x nLayers_v
        # first layer carpet
        weldNodes_h = np.arange(nNodes)[:, np.newaxis] * [1, 1] + [0, nNodes]
        if zigzag:
            weldNodes_h[:, 0] = weldNodes_h[::-1, 0]
        weldNodes_h = np.tile(weldNodes_h, (nLayers_h - 1, 1)) + np.repeat(nNodes * np.arange(nLayers_h - 1)[:, np.newaxis], nNodes, axis=0)
        ## welding TR to TL and BR to BL: weldDOF = [ [[iNode,1,iNode+nNodes,0],[iNode,3,iNode+nNodes,2]] for iNode in range(nNodes)]
        weldDOF_h = np.tile(weldNodes_h[:, [0, 0, 1, 1]] * [1, 0, 1, 0], (2, 1)) + np.repeat([[0, 1, 0, 0], [0, 3, 0, 2]], weldNodes_h.shape[0], axis=0)  # horizontal welding (next to each other)

        # following layers
        # all nLayers_v carpets
        weldNodes_h = np.tile(weldNodes_h, (nLayers_v, 1)) + np.repeat(nLayers_h * nNodes * np.arange(nLayers_v)[:, np.newaxis], (nLayers_h - 1) * nNodes, axis=0)
        weldDOF_h = np.tile(weldNodes_h[:, [0, 0, 1, 1]] * [1, 0, 1, 0], (2, 1)) + np.repeat([[0, 1, 0, 0], [0, 3, 0, 2]], weldNodes_h.shape[0], axis=0)  # horizontal welding (next to each other)

        # weld carpets to each other
        weldNodes_v = np.arange(nNodes)[:, np.newaxis] * [1, 1] + [0, nNodes * nLayers_h]
        if zigzag:
            weldNodes_v[:, 1] = weldNodes_v[::-1, 1]
        weldNodes_v = np.tile(weldNodes_v, (nLayers_h, 1)) + np.repeat(nNodes * np.arange(nLayers_h)[:, np.newaxis], nNodes, axis=0)
        ## welding TL to BL and TR to BR: weldDOF = [ [[iNode,0,iNode+nNodes,2],[iNode,1,iNode+nNodes,3]] for iNode in range(nNodes)]
        weldDOF_v = np.tile(weldNodes_v[:, [0, 0, 1, 1]] * [1, 0, 1, 0], (2, 1)) + np.repeat([[0, 0, 0, 2], [0, 1, 0, 3]], weldNodes_v.shape[0], axis=0)  # vertical welding (on top of each other)

        weldDOF = np.concatenate((weldDOF_h, weldDOF_v), axis=0)
    return weldDOF


def stacking_offset(L, Hn, nNodes, nLayers_v, offsetType="linear", tcoef=1, ncoef=1):
    """
    offset between successive layers along t and n
    Returns offset of size (nLayers_v, 2)
    """
    lc = L / (nNodes - 1)  # length of elements
    offset = np.ones((nLayers_v, 2)) * [tcoef, ncoef]
    if offsetType == "linear":
        offset *= np.linspace(0, 1, nLayers_v)[:, np.newaxis]
    elif offsetType == "exponential":
        offset *= np.exp(np.linspace(0, np.log(2), nLayers_v)[:, np.newaxis]) - 1
    elif offsetType == "sinus":
        offset *= np.sin(np.linspace(0, 2 * 2 * np.pi, nLayers_v)[:, np.newaxis])
    elif offsetType == "cosinus":
        offset *= np.cos(np.linspace(0, 2 * 2 * np.pi, nLayers_v)[:, np.newaxis])
    elif offsetType == "exp-sinus" :
        offset *= (np.exp(np.linspace(0, np.log(2), nLayers_v)[:, np.newaxis])) * np.sin(np.linspace(0, 1.5 * 2 * np.pi, nLayers_v)[:, np.newaxis])

    offset *= [lc, Hn]
    return offset


def mesh_offset(X, Elems, nNodes, nLayers_h, nLayers_v, zigzag, offset, layerType):
    """
    Function that applies a stacking offset between successive layers of the whole structure in the build direction

    Parameters
    ----------
    stacking_offset : array of size (nLayers_v, 2)
        offset between successive layers along t and n direction
    X : array of size (nNodes, 3)
        Coordinates of each node in the global reference (t,n,b).
    nNodes : int
        Number of nodes discretising the length L.
    beadType : str, optional
        Geometry of the first layer. Options available are "linear", "circular", "square", "quarter_circular", "sinus"
        The default is "linear".

    Returns
    -------
    X : array of size (nNodes*nLayers, 3)
        Coordinates of each node in the global reference (t,n,b).
    """

    ## Apply offset in the building direction
    nNodes_h = nNodes * nLayers_h

    if layerType == "duplicate":  # "linear":
        dX = np.repeat(offset, nNodes_h, axis=0)

    else:
        t, n = local_axis(X, Elems)
        if zigzag:  # if zigzag, t and n change sign at every bead, prevent this by changing signs manually
            signs = np.repeat(np.array([[1, -1]]), nNodes, axis=1).flatten()  # alternating minus signs for two consecutive beads
            signs = np.tile(signs, nLayers_h // 2)  # alternating minus signs for all horizontal beads
            if nLayers_h % 2:  # if number of beads per layer is uneven, add one last row of nNodes positive signs
                signs = np.concatenate((signs, np.ones(nNodes)), axis=0)
            signs = np.tile(signs, nLayers_v)  # alternating minus signs for all vertical beads
            signs.reshape(nLayers_v, nNodes_h)[1::2, :] *= -1  # changing signs for uneven layers since successive layers have different orientations

            t *= signs[:, np.newaxis]
            n *= signs[:, np.newaxis]

        dX = np.repeat(offset, nNodes_h, axis=0)
        dX = dX[:, 0][:, np.newaxis] * t[:, :-1] + dX[:, 1][:, np.newaxis] * n[:, :-1]

    X[:, :-1] += dX  # add offset
    return X


def second_order_discretization(X, Elems, U0):
    """
    Function that updates the mesh and discretisation in the case of second order element in the FEM definition
    Three-nodes element
    
    Parameters
    ----------
    X : array of size (nNodes*nLayers, nCoord)
        Coordinates of each node in the global reference (t,n,b).
    Elems : array of size ((nNodes - 1)*nLayers, 2)
        Index of the pair of nodes forming an element.
    U0 : array of size (3, 4, nNodes*nLayers)
        Boundary conditions for particles attached to each node : NaN when the particule has no imposed bc. Particules are indexed as {TL,TR,BL,BR}. 

    Returns
    -------
    X : array of size ((2*nNodes-1)*nLayers, nCoord)
        Coordinates of each node in the global reference (t,n,b).
    Elems : array of size ((nNodes - 1)*nLayers, 3)
        Index of the pair of nodes forming an element.
    U0 : array of size (3, 4, (2*nNodes-1)*nLayers)
        Boundary conditions for particles attached to each node : NaN when the particule has no imposed bc. Particules are indexed as {TL,TR,BL,BR}. 
    """

    # element mid-point to be added
    Xmid = np.mean(X[Elems, :], axis=1)
    # interpolate BC
    U0 = np.concatenate((U0, np.mean(U0[:, :, Elems], axis=3)), axis=2)
    # update the element nodes list
    Elems = np.hstack([Elems[:, 0][:, np.newaxis], np.arange(Elems.shape[0])[:, np.newaxis] + X.shape[0], Elems[:, 1][:, np.newaxis]])
    # update the nodes coordinates list
    X = np.vstack([X, Xmid])
    return X, Elems, U0


def uncouple_nodes(X, Elems):
    """
    Function that transforms the "coupled" vectors (X, Elems) in their uncoupled version
    Elements were numbered as such V = [[0, 1], [1, 2], ...] and now Vunc = [[0, 1], [2, 3], ...]

    Parameters
    ----------
    X : array of size (nNodes*nLayers, nCoord)
        Coordinates of each node in the global reference (t,n,b).
    Elems : array of size ((nNodes - 1)*nLayers, elemOrder + 1)
        Index of the pair of nodes forming an element.
    
    Returns
    -------
    Xunc : array of size ((elemOrder+1)*(nNodes - 2 + 1)*nLayers, nCoord) or (nUncNodes, nCoord)
        Coordinates of each uncoupled nodes in the global reference (t,n,b)..
    uncElems : array of size ((nNodes - 1)*nLayers, elemOrder + 1) or (nElemTot, elemOrder + 1)
        Index of the pair of nodes forming an uncoupled element.

    """
    nElems = Elems.shape[0]
    nNodesByElem = Elems.shape[1]

    # Coordinates of nodes of each element, as if they were uncoupled (nUncNodes,nCoord)
    Xunc = X[Elems.flatten(), :]

    # New element-uncoupled node connectivity
    uncElems = np.arange(nElems)[:, np.newaxis] * nNodesByElem + np.arange(nNodesByElem)
    return Xunc, uncElems
