""" Module containing the welding functions of the QuadWire model """
#%% import packages
import numpy as np
import scipy as sp
from scipy import sparse

#%% Added during "lab to market" : DocString + Verif test to do

def weldnode_matrices(Elems, uncElems):
    """
    Function that builds the node welding matrices 

    Parameters
    ----------
    Elems : array of size  (nElemTot, elemOrder +1)
        Index of the pair of nodes forming an element.
    uncElems : array of size or (nElemTot, elemOrder +1)
        Index of the pair of nodes forming an uncoupled element.

    Returns
    -------
    Sni : sparse matrix of shape (nUncNodes, nNodesTot)
        Welding matrix for the node functions.
    Sn : sparse matrix of shape (nUncNodes*nNodeDof, nNodesTot*nNodeDof) 
        Welding matrix for the degrees of freedom 3*4*Sni.

    """
    nElems = Elems.shape[0]
    nNodesByElem = Elems.shape[1]
    
    # Weld uncoupled nodes
    Sni = sp.sparse.csr_matrix((np.ones((nElems * nNodesByElem,)), (uncElems.flatten(), Elems.flatten())))
    
    Sn = sp.sparse.kron(sp.sparse.eye(12), Sni)
    
    return Sni, Sn

def weldwire_matrix(nNodesTot, weldDof):
    """
    Function that builds the wire welding matrices
    nNodesTOT = X.shape[0]

    Parameters
    ----------
    nNodesTot : int
        Number of total nodes discretising the structures.
    weldDOF : array of size (nNodes*(nLayers-1)*2, 4) - number of layer interface * two welding conditions
        Array listing the welding conditions between the wires 
        The k^th welding condition is given by : weldDOF[k] = [indexNode1, node1Particule, indexNode2, node2Particule] which means that the node1Particule chosen is welded to the node2Particule
        There are nInterface*nNodes*2 welding conditions in between wires.

    Returns
    -------
    Sw : sparse matrix of shape (nNodesTot*nNodeDof, nNodesTot*nNodeDof)
        Inter-wire welding matrix / wire connectivity matrix 

    """
    # There are welding conditions to apply
    if np.array(weldDof).size != 0:  
        # DOF indexes
        wDOFidx = weldDof[:, [0, 2]] + nNodesTot * weldDof[:, [1, 3]] 
        # Repeat for all 3 coordinates
        wDOFidx = np.tile(wDOFidx, (3, 1)) + np.repeat(4 * nNodesTot * np.arange(3)[:, np.newaxis], wDOFidx.shape[0], axis=0)  
        noWeld = np.argwhere(np.invert(np.isin(np.arange(12 * nNodesTot), wDOFidx.flatten())))
        # Build welding matrix
        ii = np.hstack((noWeld.flatten(), wDOFidx[:, 0], wDOFidx[:, 1]))
        jj = np.hstack((noWeld.flatten(), wDOFidx[:, 0], wDOFidx[:, 0]))
        Sw = sp.sparse.csr_matrix((np.ones(ii.shape), (ii, jj)), shape=(12 * nNodesTot, 12 * nNodesTot))
    # No welded DOFs, gives Identity matrix
    else:  
        Sw = sp.sparse.eye(12 * nNodesTot)
    
    return Sw
    
def bcs_matrix(U0, Sw):
    """
    Function that builds the dof removing matrix.

    Parameters
    ----------
    U0 : array of size (nCoord, nParticule, nNodes*nLayers)
        Boundary conditions for particles attached to each node : NaN when the particule has no imposed bc. Particules are indexed as {TL,TR,BL,BR}. 
    Sw : sparse matrix of shape (nNodesTot*nNodeDof, nNodesTot*nNodeDof)
        Inter-wire welding matrix / wire connectivity matrix.

    Returns
    -------
    Y : sparse matrix of size (nNodesTot*nNodeDof, nNodesTot*nNodeDof/2) 
        /2 because two particules are used to weld two wires
        Degrees of freedom removing matrix.

    """
    # Reshape U0 (3, 4, nNodes*nLayers) matrix as a vector
    u0 = U0.flatten()
    # True/False table storing remaining DOF
    freeDOF = np.isnan(u0)   
    # Boundary conditions matrix
    B = sp.sparse.diags(freeDOF * 1) 
    # Delete DOFs
    Y = Sw @ B
    isZero = np.sum(Y, 0) == 0
    Y = Y[:, np.array(np.invert(isZero)).flatten()]
    return Y



