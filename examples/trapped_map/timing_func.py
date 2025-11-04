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


def timing_func(Ekin_frac,neta_poinc,ns_poinc,Nmaps=1000):
    
    time1 = time.time()
    boozmn_filename = "../inputs/boozmn_equil_G1600_DESC_fixed.nc"
    Ekin = FUSION_ALPHA_PARTICLE_ENERGY*Ekin_frac
    # neta_poinc = 5  # Number of eta initial conditions for poincare
    # ns_poinc = 120  # Number of s initial conditions for poincare
    # Nmaps = 1000  # Number of Poincare return maps to compute
    modBin = 5.95
    # call_DESC = False
    tmax = 1e-2

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
    helicity_N = 0
    degree = 3  # Degree for Lagrange interpolation

    bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world, helicity_N=helicity_N, helicity_M=helicity_M) # specify helicities to filter QS-breaking modes
    # bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world,)

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

    # time2 = time.time()

    # proc0_print("poincare time: ", time2 - time1)

    # Compute frequencies
    omega_eta_prof, omega_b_prof, s_prof = poinc.compute_frequencies()
    time2 = time.time()


    return time2-time1,Ekin_frac,neta_poinc,ns_poinc

time_loop = range(0,10)
times = np.zeros(len(time_loop))
Ekin_fracs = np.zeros(len(time_loop))
neta_poincs = np.zeros(len(time_loop))
ns_poincs = np.zeros(len(time_loop))
for i in time_loop:
    print('loop')
    times[i],Ekin_fracs[i],neta_poincs[i],ns_poincs[i] = timing_func(Ekin_frac=0.1,neta_poinc=3,ns_poinc=(i+1)*10,Nmaps=1000)

np.save("times",times)
np.save("Ekin_fracs",Ekin_fracs)
np.save("neta_poincs",neta_poincs)
np.save("ns_poincs",ns_poincs)