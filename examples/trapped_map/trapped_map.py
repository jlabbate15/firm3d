import time

import numpy as np

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


# COMMON USER INPUTS #
boozmn_filename = "boozmn_new_QH_aScaling.nc"
Ekin = FUSION_ALPHA_PARTICLE_ENERGY*1
neta_poinc = 3  # Number of eta initial conditions for poincare
ns_poinc = 50  # Number of s initial conditions for poincare
Nmaps = 750  # Number of Poincare return maps to compute
modBin = 5.83
call_DESC = False
tmax = 1e-4
#######################


charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS

resolution = 48  # Resolution for field interpolation
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-8  # Tolerance for ODE solver
s_mirror = 0.2**2  # flux surface for mirroring
theta_mirror = np.pi / 2  # poloidal angle for mirroring
zeta_mirror = 0
helicity_M = 1  # helicity of field strength contours
helicity_N = -4
degree = 3  # Degree for Lagrange interpolation


# Setup logging to redirect output to file
# setup_logging(f"stdout_trapped_map_{resolution}_{comm_size}.txt")

time1 = time.time()

# print("start BRI")

bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world, helicity_N=helicity_N, helicity_M=helicity_M) # specify helicities to filter QS-breaking modes
# bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world)

# print("start IBF")

field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
    stellsym=True
)

print("start tracing")

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
    s_init=(np.linspace(0,1,ns_interp))**2
)

time2 = time.time()

proc0_print("poincare time: ", time2 - time1)

# Compute frequencies
omega_eta_prof, omega_b_prof, s_prof = poinc.compute_frequencies()

# Extra plotting: Plotting the objective function over the Poincare plot
# Plot objective function value on top of it
# if call_DESC:
#     import matplotlib.pyplot as plt
#     from desc.ResonanceOpt.TRObj_func import TrappedResonanceObj
#     import desc.io
#     import jax.numpy as jnp

#     # Run DESC objective function
#     fig, ax = plt.subplots()
#     eq = desc.io.load("../inputs/equil_G1600_DESC_fixed.h5")
#     rhos = (np.linspace(0.1,0.9,50))**(1/2) # rho = sqrt(s)
#     alphas = np.linspace(0,2*np.pi,3)
#     KE_frac = np.array([1]) #did 0.001 before
#     pitch_invs = jnp.array([modBin])
#     N=0 # QA

#     out = TrappedResonanceObj(eq,rhos,pitch_invs,KE_frac,alphas,N)
#     obj_val = out['obj'][:,0,0] # Only look at rho, for one pitch, for one energy
#     s_obj = np.linspace(0.1,0.9,len(obj_val))

#     Y = s_obj
#     X = np.linspace(0,2*np.pi,5)
#     Z = np.transpose(np.tile(obj_val, (5, 1)))
#     cs = ax.contourf(X,Y,Z,cmap='Blues')
#     fig.colorbar(cs, ax=ax)

#     ax = poinc.plot_poincare(ax=ax)
#     ax.figure.savefig('poincare_objective_overlay.png')

# if verbose and not call_DESC:
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0] = poinc.plot_poincare(ax=ax)
# ax[1].plot(omega_eta_prof/omega_b_prof,s_prof**0.5)
# ax[1].set_xlabel(r'$\omega_{\zeta}$')
# ax[1].yaxis.set_label_position("right")
# ax[1].yaxis.tick_right()
# ax[1].set_ylabel(r'$\rho$')
# ax[1].set_ylim(0.0,1.0)
# plt.savefig("poincare_omega.pdf")

np.save("data/omega_eta_FIRM3D_QH",omega_eta_prof/omega_b_prof)
np.save("data/s_prof_FIRM3D_QH",s_prof)