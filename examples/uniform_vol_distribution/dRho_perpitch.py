import time

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
)
from firm3d.field.tracing_helpers import (
    initialize_position_uniform_vol,
    initialize_velocity_uniform,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import proc0_print, setup_logging
from firm3d.util.mpi import comm_size, comm_world, verbose

from firm3d._core.util import parallel_loop_bounds

# from firm3d.field.trajectory_helpers import compute_loss_fraction
# from orbit_classification_Jens_drho import OrbitClassification


try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    verbose = comm.rank == 0
    comm_size = comm.size
except ImportError:
    comm = None
    verbose = True
    comm_size = 1

##### INPUTS
Bc_res = 20
# modBmin_global = 5.515158152204317+0.5
# modBmax_global = 7.375629705430016-0.5
modBmin_global = 6.05
modBmax_global = 6.8
boozmn_filename = "../inputs/boozmn_equil_G1600_DESC_fixed.nc"
#####


Bcs = np.linspace(modBmin_global,modBmax_global,Bc_res)
resolution = 48  # Resolution for field interpolation
nParticles = int(1e3)  # Number of particles to trace
reltol = 1e-8  # Relative tolerance for the ODE solver
abstol = 1e-8  # Absolute tolerance for the ODE solver
order = 3  # Order for radial interpolation
degree = 3  # Degree for 3d interpolation
tmax = 1e-2  # Time for integration
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution
helicity_M = 1 # Helicity of the field strength, used to distinguish ripple and barely-trapped orbits
helicity_N = 0


# Setup logging to redirect output to file
# setup_logging(f"stdout_{nParticles}_{resolution}_{comm_size}_lessave2.txt")
def delta_rho(s): # returns the total ds for one particle
    # s = np.array(poinc.s_all)
    s_max = np.max(s)
    s_min = np.min(s)
    return s_max**(0.5)-s_min**(0.5) # one for each s,eta initialization
## Setup radial interpolation
bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world)
## Setup 3d interpolation
field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
)

# points = initialize_position_uniform_vol(field, nParticles, comm=comm_world)
# points = initialize_position_uniform_vol(field, nParticles)

# Initialize points uniform in rho,alpha grid but specified with s,theta,zeta coordinates
rhos = np.sqrt(np.linspace(0.1,0.9,50)) # rho = sqrt(s)
alphas = np.linspace(0,2*np.pi,3)

s = rhos**2
theta = alphas
zeta = np.array(0)

A, B, C = np.meshgrid(s, theta, zeta, indexing='ij')
points = np.stack([A, B, C], axis=-1).reshape(-1, 3)


field.set_points(points)
B = field.modB()[:,0] # B at each point

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
vpar0 = np.sqrt(2 * Ekin / mass)

drho = np.zeros(len(Bcs))
nParticles_loop = np.zeros(len(Bcs))
j=0
for Bc in Bcs:
    print('Bc = ',Bc)

    # vpar_init = initialize_velocity_uniform(vpar0, nParticles, comm=comm_world)
    # parallel_speeds: A ``(nparticles, )`` array containing the speed in
    #                      direction of the B field for each particle.
    # v// = v sqrt(1 - B/Bc) * +/-1

    # Correct points and B for B>Bc (particles that are not trapped we don't consider). Could also do this by just setting their vpar>a trapped vpar
    filter_i=[]
    points_filtered=[]
    B_filtered=[]
    i=0
    for i, _B in enumerate(B):
        if _B>=Bc: # not trapped
            continue
        else:
            points_filtered.append(points[i,:])
            B_filtered.append(_B)
    points_filtered = np.array(points_filtered) # := (num particles,coord)
    nParticles_filtered = len(points_filtered[:,0])
    B_filtered = np.array(B_filtered)

    vpar_init = np.random.choice([-1, 1], size=nParticles_filtered) * vpar0 * np.sqrt(1 - B_filtered/Bc)

    ## Trace alpha particles in Boozer coordinates until they hit the s = 1 surface
    first, last = parallel_loop_bounds(comm, nParticles_filtered)
    for iParticle in range(first, last):
        point = np.zeros((1, 3))
        point[0, :] = points_filtered[iParticle, :]
        res_tys, res_zeta_hits = trace_particles_boozer(
            field,
            point,
            [vpar_init[iParticle]],
            tmax=tmax,
            mass=mass,
            charge=charge,
            Ekin=Ekin,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
            forget_exact_path=False,
            abstol=abstol,
            reltol=reltol,
            dt_save=1e-6
        )

        '''times, loss_frac = compute_loss_fraction(res_tys, tmin=1e-5, tmax=1e-2)
        out_dict = {
            'equilibria': boozmn_filename,
            'times': times,
            'loss_frac': loss_frac
        }
        np.save('PerPitchData/lossfrac_'+str(Bc),loss_frac[-1],allow_pickle=True)'''

        res_hit = res_zeta_hits[0]
        res_ty = res_tys[0]
        drho[j] += delta_rho(res_ty[:,1])
    nParticles_loop[j] = nParticles_filtered
    j+=1

np.save('dRhoPerPitchData2/drhos',drho/nParticles_filtered,allow_pickle=True)
np.save('dRhoPerPitchData2/nParticles_Bc',nParticles_loop,allow_pickle=True)
