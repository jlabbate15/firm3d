# Compute delta rho for each equilibria desired

import time

import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.trajectory_helpers import TrappedPoincare
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

# Can comment out these inputs below if call_DESC==False
from desc.ResonanceOpt.TRObj_func import TrappedResonanceObj
import desc.io
from desc.backend import set_device
set_device("cpu")
import jax.numpy as jnp


#######################
# COMMON USER INPUTS #
# firm3d setup
num_pitch = 1
boozmn_filenames = ["../inputs/boozmn_equil_G1600_DESC_fixed.nc"] # make sure this matches the DESC input order
desc_file = '~/codes/DESC_fork_08282025/DESC_TrappedRes/desc/ResonanceOpt/multiequil_plot/desc_out_multiequil.npy'
neta_poinc = 2  # Number of eta initial conditions for poincare
ns_poinc = 50  # Number of s initial conditions for poincare
Nmaps = 1000  # Number of Poincare return maps to compute
KE_frac_desc = np.array([1])
tmax = 1e-4
#######################

desc_out = np.load(desc_file) # dictionary of equilibria
# 'name': equilibria,
# 'Bc': pitch_invs,
# 'f_tr2_desc': value['f_tr2']
desc_fns = []
desc_Bcs = []
desc_val = []
desc_Ns = []
for key in desc_out: # loop through all DESC equilibria
    desc_fns.append(desc_out[key]['name'])
    desc_Bcs.append(desc_out[key]['Bc'])
    desc_val.append(desc_out[key]['f_tr2_desc'])
    desc_Ns.append(desc_out[key]['N'])

# Additional setup

Ekin = FUSION_ALPHA_PARTICLE_ENERGY * KE_frac_desc[0]
charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS

resolution = 48  # Resolution for field interpolation
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-8  # Tolerance for ODE solver
s_mirror = 0.5  # flux surface for mirroring
theta_mirror = np.pi / 2  # poloidal angle for mirroring
zeta_mirror = 0
helicity_M = 1  # helicity of field strength contours
degree = 3  # Degree for Lagrange interpolation


# Setup logging to redirect output to file
#setup_logging(f"stdout_trapped_map_{resolution}_{comm_size}.txt")

# s = np.array(poinc.etas_all) # := (334,1001) (particle,bounces), loops through etas first, then iterates s
def delta_rho(poinc): # returns the total ds for all particles initialized
    s = np.array(poinc.s_all)
    eta = np.array(poinc.etas_all)
    s_max = np.max(s,axis=1)
    s_min = np.min(s,axis=1)
    drho = s_max**(0.5)-s_min**(0.5) # one for each s,eta initialization
    return np.sum(drho,axis=0)


# field initialization
firm3d_dict = {}
loop=0
for boozmn_filename in boozmn_filenames:
    time1 = time.time()

    bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world)
    # bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world, helicity_N=helicity_N, helicity_M=helicity_M) # specify helicities to filter QS-breaking modes

    field = InterpolatedBoozerField(
        bri,
        degree,
        ns_interp=ns_interp,
        ntheta_interp=ntheta_interp,
        nzeta_interp=nzeta_interp,
    )

    time2 = time.time()
    proc0_print("BRI and IBF time: ", time2 - time1)

    # Particle tracing for multiple pitch inverses
    poinc_list = []
    poinc_drho_list = []
    modBins = desc_Bcs[loop]
    helicity_N = desc_Ns[loop] # =0 for QA
    for modBin in modBins:
        poinc = TrappedPoincare(
            field,
            helicity_M,
            helicity_N,
            s_mirror,
            theta_mirror,
            zeta_mirror,
            mass,
            charge,
            Ekin,
            modBin=modBin,
            ns_poinc=ns_poinc,
            neta_poinc=neta_poinc,
            Nmaps=Nmaps,
            comm=comm_world,
            solver_options={"reltol": tol, "abstol": tol, "axis": 0},
            tmax=tmax,
        )
        # poinc_list.append(poinc)
        poinc_drho_list.append(delta_rho(poinc))
    loop+=1
    name = desc_fns[loop]
    firm3d_dict[name] = {
        'delrho_val': np.sum(np.array(poinc_drho_list)) # sum over lambdas
    }

np.save("~/codes/DESC_fork_08282025/DESC_TrappedRes/desc/ResonanceOpt/multiequil_plot/firm3d_out",firm3d_dict)