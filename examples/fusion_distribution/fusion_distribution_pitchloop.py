# Basis code is copied from "fusion_distribution.py" file and then developed into a loop to examine multiple pitch angles #

import sys
import numpy as np
import time

from simsopt.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from simsopt.field.tracing import (
    trace_particles_boozer,
    MaxToroidalFluxStoppingCriterion,
)
from simsopt.field.tracing_helpers import (
    initialize_position_profile,
    initialize_velocity_uniform,
)
from simsopt.util.constants import (
    ALPHA_PARTICLE_MASS,
    ALPHA_PARTICLE_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from simsopt.util.functions import proc0_print

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    verbose = comm.rank == 0
    comm_size = comm.size
except ImportError:
    comm = None
    verbose = True
    comm_size = 1

# time1 = time.time()

resolution = 48  # Resolution for field interpolation
nParticles = 500  # Number of particles to trace
reltol = 1e-8  # Relative tolerance for the ODE solver
abstol = 1e-8  # Absolute tolerance for the ODE solver
order = 3  # Order for radial interpolation
degree = 3  # Degree for 3d interpolation
boozmn_filename = "boozmn_equil_G1600_DESC_fixed.nc"
tmax = 1e0  # Time for integration
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution

# sys.stdout = open(f"stdout_{nParticles}_{resolution}_{comm_size}.txt", "a", buffering=1)

## Setup radial interpolation
bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm) # original line
# bri = BoozerRadialInterpolant("wout_equil_Helios_E0090-13_DESC_fixed.nc", order, no_K=True, mpol=48,ntor=48, verbose=True)

## Setup 3d interpolation
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

# Define fusion birth distribution
# Bader, A., et al. "Modeling of energetic particle transport in optimized stellarators." Nuclear Fusion 61.11 (2021): 116060.
nD = lambda s: (1 - s**5)  # Normalized density
nT = nD
T = lambda s: 11.5 * (1 - s)  # Temperature in keV


# D-T cross-section
def sigmav(T):
    if T > 0:
        return T ** (-2 / 3) * np.exp(-19.94 * T ** (-1 / 3))
    else:
        return 0


# Reactivity profile
reactivity = lambda s: nD(s) * nT(s) * sigmav(T(s))


from simsopt.field.trajectory_helpers import compute_loss_fraction

## Post-process results to obtain lost particles
# if verbose:
import matplotlib

matplotlib.use("Agg")  # Don't use interactive backend
import matplotlib.pyplot as plt

pitch_invs = np.linspace(5.9,6.1,20)

points = initialize_position_profile(field, nParticles, reactivity, comm=comm)

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Initialize uniformly distributed parallel velocities
vpar0 = np.sqrt(2 * Ekin / mass)

tmin = 1e-5
tmax = 1e-2
sz = len(np.logspace(np.log10(tmin), np.log10(tmax), 1000))
loss_frac_arr = np.zeros((len(pitch_invs),sz)) # each pitch inverse row has corresponding loss frac at each time
i=0
for pitch_inv in pitch_invs:
    # Setup different pitch angles
    # vpar_init = initialize_velocity_uniform(vpar0, nParticles, comm=comm)
    vpar_init = np.ones(nParticles) * pitch_inv

    ## Trace alpha particles in Boozer coordinates until they hit the s = 1 surface
    res_tys, res_zeta_hits = trace_particles_boozer(
        field,
        points,
        vpar_init,
        tmax=tmax,
        mass=mass,
        charge=charge,
        comm=comm,
        Ekin=Ekin,
        stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        forget_exact_path=True,
        abstol=abstol,
        reltol=reltol,
    )

    # time2 = time.time()
    # proc0_print("Elapsed time for tracing = ", time2 - time1)

    times, loss_frac_arr[i,:] = compute_loss_fraction(res_tys, tmin=tmin, tmax=tmax)

    i+=1
    

# Plotting

plt.figure()
plt.plot(pitch_invs, loss_frac_arr[:,-1],label='Loss Fraction') # using last time point to plot loss fraction
# plt.xlim([1e-5, 1e-2])
# plt.ylim([1e-3, 1])
plt.xlabel("Pitch Inverse [T]")
# plt.ylabel("Fraction of lost particles")
plt.title("Fraction of Lost Particles at "+str(times[-1])+"s Compared With Objective Function")

obj = np.loadtxt('/Users/paullab/codes/DESC_fork_08282025/DESC_TrappedRes/desc/ResonanceOpt/obj_pitch_plot.txt')
# plt.plot(pitch_invs, obj/np.nanmax(obj),label='Objective Function (Normalized to 1)')
plt.legend()

plt.savefig("loss_fraction_vs_pitch.png")