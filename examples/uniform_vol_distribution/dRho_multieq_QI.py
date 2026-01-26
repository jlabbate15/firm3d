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

# Setup equilibria
import os
directory_path = 'Equilibria/QI'
# Get a list of all items (files and directories) in the specified path
files=os.listdir(directory_path)
boozmn_filenames = []
for file in files:
    if file[0:4] == 'booz':
        boozmn_filenames.append(file)
boozmn_filenames.sort()

# Bcs = np.linspace(modBmin_global,modBmax_global,Bc_res)
resolution = 48  # Resolution for field interpolation
nParticles = 1000  # Number of particles to trace
reltol = 1e-8  # Relative tolerance for the ODE solver
abstol = 1e-8  # Absolute tolerance for the ODE solver
order = 3  # Order for radial interpolation
degree = 3  # Degree for 3d interpolation
tmax = 1e-3  # Time for integration
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution
# helicity_M = 1 # Helicity of the field strength, used to distinguish ripple and barely-trapped orbits
# helicity_N = 0


# Setup logging to redirect output to file
setup_logging(f"stdout_{nParticles}_{resolution}_{comm_size}_QI3.txt")
def delta_rho(s): # returns the total ds for one particle
    # s = np.array(poinc.s_all)
    s_max = np.max(s)
    s_min = np.min(s)
    return s_max**(0.5)-s_min**(0.5) # one for each s,eta initialization
## Setup radial interpolation

Bcs = np.load('multieq_pitchinvs_QI.npy') # from DESC, := (equil,numBc)

drho = np.zeros(len(boozmn_filenames))
# nParticles_loop = np.zeros((len(boozmn_filenames),len(Bcs)))
i=0
for boozmn_filename in boozmn_filenames:
    # print(boozmn_filename)
    bri = BoozerRadialInterpolant(directory_path+'/'+boozmn_filename, order, no_K=True, comm=comm_world)
    ## Setup 3d interpolation
    field = InterpolatedBoozerField(
        bri,
        degree,
        ns_interp=ns_interp,
        ntheta_interp=ntheta_interp,
        nzeta_interp=nzeta_interp,
    )

    # points = initialize_position_uniform_vol(field, nParticles, comm=comm_world)
    # Initialize points uniform in rho,alpha grid but specified with s,theta,zeta coordinates
    rhos = np.sqrt(np.linspace(0.1,0.9,50)) # rho = sqrt(s)
    alphas = np.linspace(0,2*np.pi,3)

    s = rhos**2
    theta = alphas
    zeta = np.array(0)

    A, Bmesh, C = np.meshgrid(s, theta, zeta, indexing='ij')
    points = np.stack([A, Bmesh, C], axis=-1).reshape(-1, 3)
    field.set_points(points)
    B = field.modB()[:,0] # B at each point

    Ekin = FUSION_ALPHA_PARTICLE_ENERGY
    mass = ALPHA_PARTICLE_MASS
    charge = ALPHA_PARTICLE_CHARGE
    vpar0 = np.sqrt(2 * Ekin / mass)

    
    # # Initialize uniformly distributed parallel velocities
    # vpar_init = initialize_velocity_uniform(vpar0, nParticles, comm=comm_world)

    # Initialize parallel velocities over a uniform Bc
    drho_Bc = 0
    for Bc in Bcs[i][:]:
        # Correct points and B for B>Bc (particles that are not trapped we don't consider). Could also do this by just setting their vpar>a trapped vpar
        filter_i=[]
        points_filtered=[]
        B_filtered=[]
        iB=0
        for iB, _B in enumerate(B):
            if _B>=Bc: # not trapped
                continue
            else:
                points_filtered.append(points[iB,:])
                B_filtered.append(_B)
        points_filtered = np.array(points_filtered) # := (num particles,coord)
        # print(points_filtered)
        if len(points_filtered)<1:
            continue
        nParticles_filtered = len(points_filtered[:,0])
        B_filtered = np.array(B_filtered)

        vpar_init = np.random.choice([-1, 1], size=nParticles_filtered) * vpar0 * np.sqrt(1 - B_filtered/Bc)

        drho_pars = 0
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
            # print('here3')
            res_hit = res_zeta_hits[0]
            res_ty = res_tys[0]

            drho_pars += delta_rho(res_ty[:,1])
        drho[i]+=drho_pars/nParticles_filtered
    print('done with iteration '+str(i))
    i+=1

np.save('dRhoPerEquilData3/drhosQI_uniformed',drho,allow_pickle=True)
np.save('dRhoPerEquilData3/equilibria_inorderQI_uniformed',boozmn_filenames,allow_pickle=True)
