# QuadWire-Elastic
<div align="justify"> This project presents a new mechanical model named QuadWire dedicated to efficient simulations of bead-based additive manufacturing processes in which elongated beads undergoing significant cooling and eigenstrains are assembled to form 3D parts. The key contribution is to use a multi-particular approach containing 4 particles per material point to develop an extended 1D model capable of capturing complex 3D mechanical states, while significantly reducing computation time with respect to conventional approaches. 4 displacement fields, hence 12 degrees of freedom (DOF) per material point instead of 3 for conventional 3D models. Hence, within the framework of Finite Element Analysis (FEA), the bead height and thickness are internal dimensions of the proposed approach so that the mesh along the tangential direction can be much coarser than in 3D. Thus, although the extended model has more DOFs per node than classical 3D models, the total number of DOFs is reduced by several orders of magnitude for large parts.

The present work relies on on decoupling mechanics from the computation of eigenstrains as temperature kinetics and other eigenstrains are assumed to be known from previous computations.
 
More information about the QuadWire model can be find in the article _QuadWire: an extended one dimensional model for efficient mechanical simulations of bead-based additive manufacturing processes_.

The repository contains the finite element framework of the QuadWire model in Python. The current version relies on classical elastic mechanical behavior that can either be derived from an homogeneization procedure or from an optimization procedure (fitted on the energetical response of a 3D Cauchy medium). Two different versions are proposed : a [structure](qw_strucutre.py) version where eigenstrains are applied once the structure has been printed and an [additive](qw_additive.py) version which simulates the addition of matter in the additive manufacturing process.

**Keywords**: Additive Manufacturing ; Fast Thermo-Mechanical Analysis ; Model reduction  ; Multi-particular Model ; QuadWire Model ; Numerical implementation.

# How to use
We welcome you to get started from the [example file](examples.py) that reproduces the examples of the QuadWire article. Theese simulations provides a complete simulation in Fused Deposition Modeling of PLA is carried out to demonstrate the model capabilities on two examples : a wall and a carpet.

All the documentation on the modules and the functions implemented can be found [here](docs/_build/html/index.html) or using python's help function to access the docstings the implemented functions.


# License
This project is licensed under the GNU General Public License v3.0
