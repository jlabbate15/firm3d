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

boozmn_filename = "../inputs/boozmn_equil_G1600_DESC_fixed.nc"

KE_fracs = np.array([1e-5,1e-4,1e-3,1e-2,1e-1,4e-1,7e-1,1])
# KE_fracs = np.array([0.1]) # test

rhos = np.sqrt(np.linspace(0.1,0.9,5)) # rho = sqrt(s)

charge = ALPHA_PARTICLE_CHARGE
mass = ALPHA_PARTICLE_MASS
Ekin = FUSION_ALPHA_PARTICLE_ENERGY

resolution = 48  # Resolution for field interpolation
neta_poinc = 5  # Number of eta initial conditions for poincare
ns_poinc = 5  # Number of s initial conditions for poincare
Nmaps = 2500  # Number of Poincare return maps to compute
ns_interp = resolution  # number of radial grid points for interpolation
ntheta_interp = resolution  # number of poloidal grid points for interpolation
nzeta_interp = resolution  # number of toroidal grid points for interpolation
order = 3  # order for interpolation
tol = 1e-8  # Tolerance for ODE solver
s_mirror = 0.5  # flux surface for mirroring
theta_mirror = np.pi / 2  # poloidal angle for mirroring
zeta_mirror = 0
helicity_M = 1  # helicity of field strength contours
helicity_N = 0
degree = 3  # Degree for Lagrange interpolation

# Setup logging to redirect output to file
setup_logging(f"stdout_trapped_map_{resolution}_{comm_size}.txt")

# time1 = time.time()

bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world)

field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

omega_arr = []
s_arr = []
for KE_frac in KE_fracs:
    print(KE_frac)
    poinc = TrappedPoincare(
        field,
        helicity_M,
        helicity_N,
        s_mirror,
        theta_mirror,
        zeta_mirror,
        mass,
        charge,
        Ekin*KE_frac,
        ns_poinc=ns_poinc,
        neta_poinc=neta_poinc,
        Nmaps=Nmaps,
        comm=comm_world,
        solver_options={"reltol": tol, "abstol": tol, "axis": 0},
        tmax=1e-2,
        modBin=5.8
    )

    omega_eta_prof, omega_b_prof, s_prof = poinc.compute_frequencies()
    Omega_eta = omega_eta_prof / omega_b_prof

    omega_arr.append(Omega_eta)
    s_arr.append(s_prof)

np.save('omega_compare',np.array(omega_arr,dtype=object), allow_pickle=True)
np.save('s_compare',np.array(s_arr,dtype=object), allow_pickle=True)

# if verbose:
#     poinc.plot_poincare()

# time2 = time.time()

# proc0_print("poincare time: ", time2 - time1)