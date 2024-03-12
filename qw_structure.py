""" 
QuadWire function for calculating printed structures 
"""
#%% Imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from modules import mesh, fem, weld, behavior, plot

#%% Function

def structure(L, Hn, Hb, meshType, nNodes, nLayers, buildDirection, dT, loadType, elemOrder, quadOrder, optimizedBehavior, toPlot, clrmap="stt", scfplot = 10 ):
    """
    Implementation of the mechanical calculation core of the QuadWire model for a pre-printed structure.
    Wires are odered  :   TL -- TR
                          |     |
                         BL -- BR
    Particules in the article are enumerated such as [TL,TR,BL,BR] = [3,1,4,2] 
    Primary variables: U_alpha = [[uTL_t,uTR_t,uBL_t,uBR_t],[uTL_n,uTR_n,uBL_n,uBR_n],[uTL_b,uTR_b,uBL_b,uBR_b]]
    
    Parameters
    ----------
    L : int
         Length of the structure in tangential direction t.
    Hn : TYPE
        DESCRIPTION.
    Hb : TYPE
        DESCRIPTION.
    meshType : str, optional
        Geometry of the first layer. Options available are "linear", "circular" or "sinus". 
        The default is "linear". Other options are "circular" or "sinus".
    nNodes : int
        Number of nodes discretising the length L.
    nLayers : int
        Number of layers of the structure in the build dirction.
    buildDirection : str
        Build direction of the structure.
        Options are : "v" for a wall structure and "h" for a carpet structure.
    dT : float
       Temperature variation
    loadType : str
        Type of thermal loading.
        Options are "uniform", "linear", "quad" or "random"
     elemOrder : int
         Element order : number of nodes that forms an element 
         Options are 1 for two-nodes P1 element and 2 for three-nodes P2 element
    quadOrder : int
        Element quadrature order.
    optimizedBehavior : bool
        If True, the material behavior used is obtained through an optimisation procedure as described in the article.
        If False, the material behavior used is obtained through an homogeneization procedure.
    toPlot : bool
        If True, several outputs are ploted.
        If False, no outputs are ploted
    clrmap : str, optional
        Option for the plot function.
        Choice of the quantity displayed in the colormap of the printed structure.
        The default is "stt". Options are "stt" for sigma_tt and "nrg" for the energy
    scfplot : int, optional
        Aspect ratio between the length and height (or width) of the section. 
        Used for display functions. 
        The default is 10.

    Returns
    -------
    U : array of float of size (nCoord, nParticules, nNodesTOT) 
        Nodal displacement result.
        Coordinates are ordered [t, n, b]
        Particules are ordered [TL,TR,BL,BR] = [3,1,4,2] 
        Nodes are ordered by layers.
    Eps : array of size (nInc*nQP, 1)
        QW generalized strains.
        k*nQP to (k+1)*nQP gives the k^th generalized strain at each quadrature point
    Sigma : array of size (nInc*nQP, 1)
        QW generalized stresses.
        k*nQP to (k+1)*nQP gives the k^th generalized stress at each quadrature point
    nrg : array of size (nElemsTOT,)
        Linear elastic energy density of each elements.
        Elements are ordered by layer.

    """
    
    ### Discretization and meshing
    X, Elems, U0 = mesh.meshing_first_layer(L, nNodes, meshType, False) 
    X, Elems, U0 = mesh.mesh_structure(X, Elems, U0, nLayers, nNodes, buildDirection, Hb)
    weldDof = mesh.welding_conditions(X, Elems, U0, nLayers, nNodes, buildDirection, Hb)

    if elemOrder==2:
        X, Elems, U0 = mesh.second_order_discretization(X, Elems, U0)

    Xunc, uncElems = mesh.uncouple_nodes(X, Elems)


    ### Prevision taille
    nElemsTOT = Elems.shape[0]
    nUncNodes = Xunc.shape[0]
    nNodesTOT = X.shape[0]
    nParticules = 4
    nCoord = 3
    nNodeDOF = nCoord * nParticules 

    ### Usefull matrices
    ## Integration matrices
    xiQ, wQ = fem.element_quadrature(quadOrder)
    XIQ, WQ = fem.fullmesh_quadrature(quadOrder, nElemsTOT)
    nQP = WQ.shape[0]
    N, Dxi, Ds, J, W, O, qp2elem, elemQ = fem.integration_matrices(X, Elems, elemOrder, quadOrder) 
    ## Identity matrix (nQP, nQP)
    I_nqp = sp.sparse.diags(np.ones(nQP)) 
    ## Projection matrices
    T, Tv, Tc, Ta = fem.alpha2beta_matrices(Hn, Hb, nNodesTOT, nUncNodes)
    t, n, b, P = fem.local2global_matrices(Xunc, uncElems, elemOrder)
    ## Welding matrices
    Sni, Sn = weld.weldnode_matrices(Elems, uncElems)
    Sw = weld.weldwire_matrix(nNodesTOT, weldDof)
    Y = weld.bcs_matrix(U0, Sw)
    # matrice de passage noeud alpha -> qp beta
    Assemble = Ta @ P.T @ Sn  

    ### Behavior
    ## PLA material parameters
    E = 3e3  # Young modulus /!\ in MPa !n
    nu = 0.3  # poisson ratio
    alpha = 1.13e-5  # thermal expansion coefficient
    k0 = E / (3 * (1 - 2 * nu))  # bulk modulus (K)
    mu = E / (2 * (1 + nu))  # shear modulus (G)
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  # k0-2/3*mu
    
    ## Local behavior
    Btot = behavior.derivation_xi(Hn, Hb, N, Ds)
    Ctot = behavior.derivation_chi(Hn, Hb, N, Ds)

    if optimizedBehavior :
        Rxi, Rchi = behavior.optimization_Rxi_Rchi()
    else : 
        Rxi = behavior.homogeneization_Rxi(Hn, Hb, lmbda, mu)
        Rchi = behavior.homogenization_Rchi(L, Hn, Hb, lmbda, mu, nNodes)
    
    ## Local stiffness matrix
    Kxi = behavior.assembly_behavior(Rxi, Btot, W)
    Kchi = behavior.assembly_behavior(Rchi, Ctot, W)
    K = Kxi + Kchi

    ### Thermal eigenstrain
    dTalpha = behavior.dTfcn(N @ Xunc, dT, loadType, nLayers)
    dTbeta = dTalpha @ T
    dTmoy = dTbeta[:, 0][:, np.newaxis]

    ### Assembly of K.u = f
    # hard copy is needed to keep the u0 vector unmodified
    u00 = U0.flatten()[:, np.newaxis].copy()  
    freeDOF = np.isnan(U0.flatten())  # True/False table storing remaining DOF
    u00[freeDOF] = 0  # cull NaNs

    ## Construction of the global stiffness matrix
    # K = Ta.T @ K @ Ta # Projection alpha -> beta puis beta -> alpha
    # K = P @ K @ P.T    # Projection global -> local puis local -> global
    # K = Sn.T @ K @ Sn  # Welding nodes
    K = Assemble.T @ K @ Assemble
    yKy = Y.T @ K @ Y  # Deleting non usefull dof

    ## Construction of the force vector
    # Thermal eigenstrain
    Eps_thermal = behavior.thermal_eigenstrain(alpha, dTmoy)
    f_thermal = Btot.T @ sp.sparse.kron(Rxi, W) @ Eps_thermal
    # Assembly of the force vector
    f = np.zeros(nUncNodes * nNodeDOF)
    f = f_thermal
    f = Assemble.T @ f
    fbc = f - K @ u00 
    yfbc = Y.T @ fbc  # Deleting non usefull dof


    ### Solve K.u = f
    us = sp.sparse.linalg.spsolve(yKy, yfbc)

    ### Reconstruction of the problem's unknowns
    ## Displacement field
    u = Y @ us[:, np.newaxis]
    U = u.reshape((3, 4, nNodesTOT))
    # nodal displacement 
    uUncouple = Assemble @ u  # beta configuration
    uUncouple_alpha =  P.T @ Sn @ u # alpha configuration

    ## Strains
    # Initialization
    Eps = np.zeros(((6 + 9) * nQP, 1)) # (nInc, nQP) sigma_xi = Rxi @ (Eps_xi - Eps_thermique)
    Eps_xi = np.zeros((6*nQP, 1))
    Eps_chi = np.zeros((9*nQP, 1))
    # Update
    Eps += sp.sparse.vstack((Btot, Ctot)) @ uUncouple  
    Eps_xi += Btot @ uUncouple
    Eps_chi += Ctot @ uUncouple
    
    
    ## Stresses
    # Initialization
    Sigma = np.zeros(((6 + 9) * nQP, 1))  # 6 strain components and 9 curvature components
    Sigma_s = np.zeros((6*nQP, 1))
    Sigma_m = np.zeros((9*nQP, 1))
    # Rtot : Mauvais nom
    Rtot = sp.sparse.block_diag((sp.sparse.kron(Rxi, I_nqp), sp.sparse.kron(Rchi, I_nqp)))
    # Update
    Sigma +=  Rtot @ Eps
    Sigma -=  sp.sparse.vstack((sp.sparse.kron(Rxi, I_nqp) @ Eps_thermal, sp.sparse.csr_matrix((9 * nQP, 1)) ))
    Sigma = np.array(Sigma) ## changement de type de variable donc il faut la repasser en ndarray wtf ?

    Sigma_s += sp.sparse.kron(Rxi, I_nqp) @ (Eps_xi - Eps_thermal)
    Sigma_m +=  sp.sparse.kron(Rchi, I_nqp) @ Eps_chi

    # Sigmatt elem
    sigmatt_elem = qp2elem @ Sigma[:nQP]

    ### Energy
    Eps_th = sp.sparse.vstack((Eps_thermal, sp.sparse.csr_matrix((9*nQP, 1)))).toarray()

    qpEnergyDensity = behavior.energyDensity(Eps, Eps_th, Sigma, Rtot, quadOrder, 15)
    nrg = qp2elem @ qpEnergyDensity
    
    ### Plot
    
    if toPlot :
        fig = plt.figure()
        ax = plt.axes(projection='3d', proj_type='ortho')
        # ax.set_axis_off()

        # mean basis vectors on nodes
        Me = sp.sparse.csr_matrix((np.ones(Elems.size), (Elems.flatten(), np.arange(Elems.size))))
        nm = Me @ n;
        nm = nm / np.linalg.norm(nm, axis=1)[:, np.newaxis]
        bm = Me @ b;
        bm = bm / np.linalg.norm(bm, axis=1)[:, np.newaxis]

        # undeformed shape
        x = X[:, :, np.newaxis] + 0.5 * (
                Hn * nm[:, :, np.newaxis] * np.array([[[-1, 1, -1, 1]]]) + Hb * bm[:, :, np.newaxis] * np.array(
            [[[1, 1, -1, -1]]]))
        # plotMesh(x,Elems,color='none',edgecolor='gray')
        ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1]*scfplot), np.ptp(x[:, 2])*scfplot))

        # deformed shape
        scale = .25 * np.linalg.norm([Hn, Hb]) / np.max(abs(U), (0, 1, 2))
        uplot = np.moveaxis(U, (0, 1, 2), (1, 2, 0))
        x = x + scale * uplot
        
        if clrmap=="stt":
            clr = sigmatt_elem[:,0]
            # sigmatt = sp.sparse.linalg.lsqr(N @ Sni, Sigma[0:nQP])[0][:, None]
            # clr = sigmatt[:, None] * [[1, 1, 1, 1]]
        elif clrmap=="nrg" :
            clr = nrg
        # clr = sigmatt_elem[:,0]
        # srf.set_array(np.mean(clr.flatten()[tri], axis=1))
        # srf.set_clim(np.nanmin(clr), np.nanmax(clr))  

        srf, tri = plot.plotMesh(ax, L, x, Elems, color='none', edgecolor='none', clrfun=clr  , outer=False)
        colorbar = plt.colorbar(srf, pad=0.15)

        if clrmap=="stt":
            colorbar.set_label('$\Sigma_{tt}$ [N]')
        elif clrmap=="nrg" :
            colorbar.set_label("$\psi$ [J.mm$^{-1}$]")

        ax.set_xlabel('Axe t')
        ax.set_ylabel('Axe n')
        ax.set_zlabel('Axe b')

        plt.show()
    
    return U, Eps, Sigma, nrg