""" 
QuadWire function for culculation during printing 
"""
#%% Imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from modules import mesh, fem, weld, behavior, plot, thermaldata, forces

#%% Function

def additive(path, L, Hn, Hb, meshType, nNodes, nLayers, buildDirection, elemOrder, quadOrder, optimizedBehavior, toPlot, clrmap="stt", scfplot = 10 ):

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
    nDOF = nNodeDOF * nUncNodes

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
    ## Assembly matrix node_alpha to qp_beta
    Assemble = Ta @ P.T @ Sn 
    elem2node = fem.elem2node(nNodes, nLayers)

    
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

    ### Thermal eigenstrain
    Tbuild = 353.15
    Tg = 328
    Tsub = 323.15
    ## Time
    tau_final =  np.int(np.ceil((nNodes-1)*3/4)) # 0 #  10 # 0 #
    nPas = (nNodes-1)*nLayers + tau_final
    
    ## Chemin des donnees thermique
    data_path = [path + '/temp_' + str(k) + ".txt" for k in range(1,nPas+1)]
    
    ## Appeler la fonction pour lire les fichiers
    Telem = thermaldata.get_data2tamb(data_path, nNodes-1, nLayers, Tbuild, Tsub)
    # Compute the dT_n = T_n - T_n-1
    dTelem = thermaldata.delta_elem_transition_vitreuse(Telem, Tbuild, Tg) 
    nPas = dTelem.shape[0]    


    # mean basis vectors on nodes
    Me = sp.sparse.csr_matrix((np.ones(Elems.size), (Elems.flatten(), np.arange(Elems.size))))
    nm = Me @ n;
    nm = nm / np.linalg.norm(nm, axis=1)[:, np.newaxis]
    bm = Me @ b;
    bm = bm / np.linalg.norm(bm, axis=1)[:, np.newaxis]
    
    # undeformed shape
    x = X[:, :, np.newaxis] + 0.5 * (Hn * nm[:, :, np.newaxis] * np.array([[[-1, 1, -1, 1]]]) + Hb * bm[:, :, np.newaxis] * np.array(
        [[[1, 1, -1, -1]]]))
    
    ### Plot initialization 
    if toPlot :
        fig = plt.figure()
        ax = plt.axes(projection='3d', proj_type='ortho')
        ax.set_box_aspect((np.ptp(x[:, 0]), np.ptp(x[:, 1]*scfplot), np.ptp(x[:, 2]*scfplot)))
        y = x 
        
        srf, tri = plot.plotMesh(ax, L, y, Elems, color='none', edgecolor='none', outer=False)
        plt.show()
    
    
    ### Initialization of the QW variables
    ## Generalised strains
    Eps = np.zeros(((6 + 9) * nQP, 1)) # (nInc, nQP) sigma_xi = Rxi @ (Eps_xi - Eps_thermique)
    Eps_xi = np.zeros((6*nQP, 1))
    Eps_chi = np.zeros((9*nQP, 1))
    
    ## Displacement
    u00 = U0.flatten()[:, np.newaxis].copy()  
    freeDOF = np.isnan(U0.flatten())  # True/False table storing remaining DOF
    u00[freeDOF] = 0  # cull NaNs
    
    V = u00
    U = u00.reshape((3, 4, nNodesTOT))
    
    ## Generalised stresses
    Sigma = np.zeros(((6 + 9) * nQP, 1))  # 6 strain components and 9 curvature components
    Sigma_time = np.zeros((nPas, (6+9)*nQP, 1))
    Sigma_s = np.zeros((6*nQP, 1))
    Sigma_m = np.zeros((9*nQP, 1))
    
    ## Stiffness zero
    tol = 1e-9
    
    ## Node deposition time
    nodeDeposTime =0.99 * np.arange(0,nNodesTOT,1) #  -20 * np.ones(nNodes)  #
  
    ### Time loop
    for actualTime in np.arange(nPas):
        ## Activated element ?
        Telem_instant = N @ Telem[actualTime][:,np.newaxis][:,[0,0]].flatten()
        activeFunction = tol + (1 - tol) * 1 / (1 + np.exp(1 * (Telem_instant - Tg)))  # size (nQP,)
        # matrice W pondérée par activeFunction (Integral weight matrix including local jacobian)
        Wa = sp.sparse.diags((WQ * J[:, np.newaxis]).flatten() * activeFunction)  
        dTelemmoy = N @ dTelem[actualTime][:,np.newaxis][:,[0,0]].flatten()[:,np.newaxis]

        ## Assembling behaviour
        Kxi = behavior.assembly_behavior(Rxi, Btot, Wa)
        Kchi = behavior.assembly_behavior(Rchi, Ctot, Wa)
        K = Kxi + Kchi
        K = Assemble.T @ K @ Assemble
        yKy = Y.T @ K @ Y  # Deleting non usefull dof

        ## Force vector 
        f = np.zeros((nUncNodes * nNodeDOF, 1))
        Eps_thermal = behavior.thermal_eigenstrain(alpha, dTelemmoy)
        f_thermal = Btot.T @ sp.sparse.kron(Rxi, Wa) @ Eps_thermal
        
        f = f_thermal
        f = Assemble.T @ f
        fbc = f - K @ u00 
        yfbc = Y.T @ fbc  # Deleting non usefull dof

        ## Solve
        vs = sp.sparse.linalg.spsolve(yKy, yfbc)
        v = Y @ vs[:, np.newaxis]
        vUncouple = Assemble @ v  # node beta configuration
        
        ## Updating displacement vector
        V = V + v
        U = U + v.reshape((3, 4, nNodesTOT))
        
        ## Updating generalized strains
        dEps = sp.sparse.vstack((Btot, Ctot)) @ vUncouple
        dEps_xi = Btot @ vUncouple
        dEps_chi = Ctot @ vUncouple
        Eps += dEps  
        Eps_xi += dEps_xi
        Eps_chi += dEps_chi
        
        ## Updating generalized stresses
        A = sp.sparse.diags(activeFunction) # activation matrix
        Rtot = sp.sparse.block_diag((sp.sparse.kron(Rxi, A), sp.sparse.kron(Rchi, A)))
        
        dSigma =  Rtot @ dEps - sp.sparse.vstack((sp.sparse.kron(Rxi, A) @ Eps_thermal, sp.sparse.csr_matrix((9 * nQP, 1)) ))
        dSigma_s = sp.sparse.kron(Rxi, A) @ (dEps_xi - Eps_thermal)
        dSigma_m =  sp.sparse.kron(Rchi, A) @ dEps_chi
        
        Sigma += dSigma
        Sigma_s += dSigma_s
        Sigma_m += dSigma_m
        Sigma_time[actualTime] = Sigma
        
        ## Plot deformed shape and stress
        if toPlot :
            if clrmap == "stt" :
                sigmaplot = sp.sparse.linalg.lsqr(N @ Sni, Sigma[0:nQP])[0][:, None]
            elif clrmap == "stn" :
                sigmaplot = sp.sparse.linalg.lsqr(N @ Sni, Sigma[nQP*3:nQP*4])[0][:, None]
            elif clrmap == "stb" :
                sigmaplot = sp.sparse.linalg.lsqr(N @ Sni, Sigma[nQP*4:nQP*5])[0][:, None]
            
            clr = sigmaplot[:, None] * [[1, 1, 1, 1]]
            clr = clr / (Hn * Hb)

            ## Update deformed shape
            scale = 0.1 * np.linalg.norm([Hn, Hb]) / np.max(abs(U), (0, 1, 2))
            uplot = np.moveaxis(U, (0, 1, 2), (1, 2, 0))
            y = x + scale * uplot
            y = np.moveaxis(y, (0, 1, 2), (0, 2, 1)).reshape(4 * y.shape[0], 3)
            srf.set_verts(y[tri])
            
            ## Plot stress
            isDeposed = actualTime*nNodesTOT/nElemsTOT > nodeDeposTime

            clr[np.invert(isDeposed)] = np.nan
            srf.set_array(np.mean(clr.flatten()[tri], axis=1))
            srf.set_clim(np.nanmin(clr), np.nanmax(clr))   
        
            plt.pause(0.005)
            
        print('actualTime:', actualTime)

    ### Plot parameters
    if toPlot :
        ax.set_xlabel('Axe t')
        ax.set_ylabel('Axe n')
        ax.set_zlabel('Axe b')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        colorbar = plt.colorbar(srf, pad=0.15)
        
        if clrmap == "stt" :
            colorbar.set_label('$\sigma_{tt}$ [MPa]')
        elif clrmap == "stn" :
            colorbar.set_label('$\sigma_{tn}$ [MPa]')
        elif clrmap == "stb" :
            colorbar.set_label('$\sigma_{tb}$ [MPa]')
    
    ###  Reconstruct internal forces
    f1, f2, f3, f4, F1, F2, F3, F4 = forces.internal_forces(Sigma, Hn, Hb)

    ### Energy
    Eps_th = sp.sparse.vstack((Eps_thermal, sp.sparse.csr_matrix((9*nQP, 1)))).toarray()
    qpEnergyDensity = behavior.energyDensity(Eps, Eps_th, Sigma, Rtot, quadOrder, 15)
    elementEnergyDensity = qp2elem @ qpEnergyDensity   

    return U, Eps, Sigma, elementEnergyDensity, qp2elem, nQP, x, Elems