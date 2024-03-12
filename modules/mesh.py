""" Module containing the discretisation and meshing functions of the QuadWire program """

#%% import packages

import numpy as np

#%% Discretization and meshing

def meshing_first_layer(L, nNodes, meshType="linear", meshRefined = False):
    """
    Function that meshes the first layer of the structure.

    Parameters
    ----------
    L : int
        Length of the structure in tangential direction t.
    nNodes : int
        Number of nodes discretising the length L.
    meshType : str, optional
        Geometry of the first layer. Options available are "linear", "circular" or "sinus". 
        The default is "linear".
    meshRefined : bool, optional
        Option available for the a "linear" mesh : refinement of the mesh close to the edges of the structure.
        The default is False.

    Returns
    -------
    X : array of size (nNodes, nCoord)
        Coordinates of each nodes in the global reference (x1, x2, x3).
    Elems : array of size (nNodes - 1, 2)
        Index of the pair of nodes forming an element.
    U0 : array of size (nCoord, nParticule, nNodes)
        Boundary conditions for particles attached to each node : NaN when the particule has no imposed bc. 
        Particules are indexed as {TL,TR,BL,BR}. 

    """
    if meshType == "linear":
        x1 = np.linspace(-1, 1, nNodes) * L / 2  # uniform mesh
        if meshRefined :
            p = 2 ; x1 = np.tanh(np.linspace(-p,p,nNodes))/np.tanh(p)*L/2 ; # heterogeneous mesh
        X = x1[:, np.newaxis] * [1, 0, 0]  # straight line along X1
        Elems = np.arange(nNodes - 1)[:, np.newaxis] + [0, 1]  # simple elements
        U0 = np.tile([[[np.nan],[np.nan],[0],[0]]],(3,1,nNodes)) # clamp all {BL,BR} wires

    elif meshType == "circular":
        theta = np.arange(nNodes)[:,np.newaxis]/nNodes*2*np.pi
        R = L/2/np.pi
        X = np.hstack((R*np.cos(theta),R*np.sin(theta),theta*0)) # straight line along X1
        Elems = np.mod(np.arange(nNodes)[:,np.newaxis]+[0,1], nNodes) # simple elements
        U0 = np.tile([[[np.nan],[np.nan],[0],[0]]],(3,1,nNodes)) # clamp all {BL,BR} wires
        
    elif meshType=="square":
        if nNodes%4 != 0 :
            raise ValueError("Le nombre de noeuds par couche renseigné n'est pas divisible par 4")
        dl = L / nNodes
        c = nNodes//4
        X = np.zeros((nNodes, 3))

        for i in range(4):
            for j in range(c):
                if i==0:
                    X[i*c + j] = [j*dl, 0, 0]
                elif i==1:
                    X[i*c + j] = [c*dl, j*dl, 0]
                elif i==2:
                    X[i*c + j] = [(c-j)*dl, c*dl, 0]
                elif i==3:
                    X[i*c + j] = [0, (c-j)*dl, 0]
                    
        Elems = np.mod(np.arange(nNodes)[:,np.newaxis]+[0,1], nNodes) # simple elements
        U0 = np.tile([[[np.nan],[np.nan],[0],[0]]],(3,1,nNodes)) # clamp all {BL,BR} wires

    elif meshType == "quarter_circular":
        t_val = 41
        theta = np.linspace(np.pi/2, np.pi/2 - (2*np.pi*t_val/360), nNodes )[:,np.newaxis]
        # theta = np.arange(nNodes)[:,np.newaxis]/nNodes* (2*np.pi*t_val/360)
        R = L/2/np.pi
        X = np.hstack((R*np.cos(theta),R*np.sin(theta),theta*0)) # straight line along X1
        Elems = np.arange(nNodes - 1)[:, np.newaxis] + [0, 1]  # simple elements
        # U0 = np.stack(( np.tile([[[0],[0],[0],[0]]],(3,1,1)) , np.tile([[[np.nan],[np.nan],[np.nan],[np.nan]]],(3,1,nNodes-1))), axis = 2) 
        U0 = np.tile([[[np.nan],[np.nan],[np.nan],[np.nan]]], (3,1,nNodes)) # glisse
        # U0 = np.tile([[[np.nan],[np.nan],[0],[0]]], (3,1,nNodes)) # plan

        U0[:, :, -1 ] = np.tile([[[0],[0],[0],[0]]],(3,1)).reshape((3,4))
        
        
    elif meshType == "sinus": 
        x1 = np.linspace(0,L,nNodes)[:,np.newaxis]
        X = np.hstack((x1,.25*L*np.sin(1.0*x1/L*2*np.pi),x1*0))
        Elems = np.arange(nNodes-1)[:,np.newaxis]+[0,1] # simple elements
        U0 = np.tile([[[np.nan],[np.nan],[0],[0]]],(3,1,nNodes)) # clamp all {BL,BR} wires
        
    else :
        raise ValueError("La maillage renseigné n'est pas connu")
    return X, Elems, U0

def mesh_structure(X, Elems, U0, nLayers, nNodes, buildDirection, H):
    """
    Function that replicates the mesh of the first layer to generate the mesh of the whole structure

    Parameters
    ----------
    X : array of size (nNodes, 3)
        Coordinates of each nodes in the global reference (t,n,b).
    Elems : array of size (nNodes - 1, 2)
        Index of the pair of nodes forming an element.
    U0 : array of size (3, 4, nNodes)
        Boundary conditions for particles attached to each node : NaN when the particule has no imposed bc. 
        Particules are indexed as {TL,TR,BL,BR}. 
    nLayers : int
        Number of layers of the structure in the build dirction.
    nNodes : int
        Number of nodes discretising the length L.
    buildDirection : str
        Build direction of the structure "v" for a wall structure and "h" for a carpet structure.
    H : float
        Width of the section of the bead in the build direction. Hb for option "v" and Hn for option "h".

    Returns
    -------
    X : array of size (nNodes*nLayers, 3)
        Coordinates of each nodes in the global reference (t,n,b).
    Elems : array of size ((nNodes - 1)*nLayers, 2)
        Index of the pair of nodes forming an element.
    U0 : array of size (3, 4, nNodes*nLayers)
        Boundary conditions for particles attached to each node : NaN when the particule has no imposed bc. Particules are indexed as {TL,TR,BL,BR}. 
    """
    ## Replicate the bead in the building direction
    if buildDirection == "h" :
        X = np.tile(X,(nLayers,1)) + np.repeat(np.arange(nLayers)[:,np.newaxis]*[0,H,0],nNodes,axis=0)             
        U0 = np.concatenate((U0,np.tile(U0,(1,1,nLayers-1))),axis=2)

    elif buildDirection == "v" :
        X = np.tile(X, (nLayers, 1)) + np.repeat(np.arange(nLayers)[:, np.newaxis] * [0, 0, H], nNodes, axis=0)  
        U0 = np.concatenate((U0, np.tile(np.nan * U0, (1, 1, nLayers - 1))), axis=2)  
            
    ## Replicate the elements
    Elems = np.tile(Elems, (nLayers, 1)) + nNodes * np.repeat(np.arange(nLayers)[:, np.newaxis], Elems.shape[0], axis=0)
    
    return X, Elems, U0


def welding_conditions(X, Elems, U0, nLayers, nNodes, buildDirection, H):
    """
    Function that returns a matrix of wire welding conditions of the structure
    
    Parameters
    ----------
    X : array of size (nNodes, 3)
        Coordinates of each nodes in the global reference (t,n,b).
    Elems : array of size (nNodes - 1, 2)
        Index of the pair of nodes forming an element.
    U0 : array of size (3, 4, nNodes)
        Boundary conditions for particles attached to each node : NaN when the particule has no imposed bc. 
        Particules are indexed as {TL,TR,BL,BR}. 
    nLayers : int
        Number of layers of the structure in the build dirction.
    nNodes : int
        Number of nodes discretising the length L.
    buildDirection : str
        Build direction of the structure "v" for a wall structure and "h" for a carpet structure.
    H : float
        Width of the section of the bead in the build direction. Hb for option "v" and Hn for option "h".

    Returns
    -------
    weldDOF : array of size (nNodes*(nLayers-1)*2, 4) - number of layer interface * two welding conditions
        Array listing the welding conditions between the wires 
        The k^th welding condition is given by : weldDOF[k] = [indexNode1, node1Particule, indexNode2, node2Particule] which means that the the node1Particule chosen is welded to the node2Particule
        There are nInterface*nNodes*2 welding conditions in between wires
    """
    
    weldNodes = np.arange(nNodes)[:,np.newaxis]*[1,1] + [0,nNodes]
    weldNodes = np.tile(weldNodes, (nLayers - 1, 1)) + np.repeat(nNodes * np.arange(nLayers - 1)[:, np.newaxis], nNodes, axis=0)

    if buildDirection == "h" :
        ## welding TR to TL and BR to BL: weldDOF = [ [[iNode,0,iNode+nNodes,2],[iNode,1,iNode+nNodes,3]] for iNode in range(nNodes)]
        weldDOF = np.tile(weldNodes[:,[0,0,1,1]]*[1,0,1,0],(2,1)) + np.repeat([[0,1,0,0],[0,3,0,2]],weldNodes.shape[0],axis=0)

    elif buildDirection == "v" :
        ## welding TL to BL and TR to BR: weldDOF = [ [[iNode,0,iNode+nNodes,2],[iNode,1,iNode+nNodes,3]] for iNode in range(nNodes)]
        weldDOF = np.tile(weldNodes[:, [0, 0, 1, 1]] * [1, 0, 1, 0], (2, 1)) + np.repeat([[0, 0, 0, 2], [0, 1, 0, 3]],
                        weldNodes.shape[0], axis=0)  # vertical welding (on top of eachother)
    return weldDOF

def second_order_discretization(X, Elems, U0):
    """
    Function that updates the mesh and discretisation in the case of second order element in the FEM definition
    Three-nodes element
    
    Parameters
    ----------
    X : array of size (nNodes*nLayers, nCoord)
        Coordinates of each nodes in the global reference (t,n,b).
    Elems : array of size ((nNodes - 1)*nLayers, 2)
        Index of the pair of nodes forming an element.
    U0 : array of size (3, 4, nNodes*nLayers)
        Boundary conditions for particles attached to each node : NaN when the particule has no imposed bc. Particules are indexed as {TL,TR,BL,BR}. 

    Returns
    -------
    X : array of size ((2*nNodes-1)*nLayers, nCoord)
        Coordinates of each nodes in the global reference (t,n,b).
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
        Coordinates of each nodes in the global reference (t,n,b).
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

