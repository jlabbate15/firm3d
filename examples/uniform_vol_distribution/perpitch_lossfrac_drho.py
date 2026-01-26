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

# time1 = time.time()


##### INPUTS
Bc_res = 10
modBmin_global = 5.515158152204317+0.5
modBmax_global = 7.375629705430016-0.5
boozmn_filename = "../inputs/boozmn_equil_G1600_DESC_fixed.nc"
#####


Bcs = np.linspace(modBmin_global,modBmax_global,Bc_res)
resolution = 48  # Resolution for field interpolation
nParticles = Bc_res*1000  # Number of particles to trace
reltol = 1e-8  # Relative tolerance for the ODE solver
abstol = 1e-8  # Absolute tolerance for the ODE solver
order = 3  # Order for radial interpolation
degree = 3  # Degree for 3d interpolation
tmax = 1e-2  # Time for integration
ns_interp = resolution
ntheta_interp = resolution
nzeta_interp = resolution



# Setup logging to redirect output to file
setup_logging(f"stdout_{nParticles}_{resolution}_{comm_size}.txt")

def delta_rho(s): # returns the total ds for one particle
    # s = np.array(poinc.s_all)
    s_max = np.max(s)
    s_min = np.min(s)
    drho = s_max**(0.5)-s_min**(0.5) # one for each s,eta initialization
    return np.sum(drho)

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

points = initialize_position_uniform_vol(field, nParticles)
field.set_points(points)
B = field.modB()[:,0] # B at each point

Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
# Initialize uniformly distributed parallel velocities
vpar0 = np.sqrt(2 * Ekin / mass)



points_filtered = []
B_filtered = []
vpar_init = []
for Bc in Bcs:
    print('Bc = ',Bc)

    vpar_init_og = initialize_velocity_uniform(vpar0, nParticles, comm=comm_world)
    # parallel_speeds: A ``(nparticles, )`` array containing the speed in
    #                      direction of the B field for each particle.
    # v// = v sqrt(1 - B/Bc) * +/-1

    # Correct points and B for B>Bc (particles that are not trapped we don't consider). Could also do this by just setting their vpar>a trapped vpar
    filter_i=[]
    points_filtered=[]
    B_filtered=[]
    i=0
    for _B in B:
        if _B>=Bc: # not trapped
            # filter_i.append(i)
            continue
        else:
            points_filtered.append(points[i,:])
            B_filtered.append(_B)
        i+=1
    # new points array is sized nParticles-len(filter_i)
    # print(np.array(points_filtered).shape)
    # print(points_filtered)
    points_filtered = np.array(points_filtered) # := (num particles,coord)
    nParticles_filtered = len(points_filtered[:,0])
    B_filtered = np.array(B_filtered)


    vpar_init = np.random.choice([-1, 1], size=nParticles_filtered) * vpar0 * np.sqrt(1 - B_filtered/Bc)
    # print(vpar_init)
    # print(vpar_init.shape)


    ## Trace alpha particles in Boozer coordinates until they hit the s = 1 surface
    print('tracing')
    res_tys, res_zeta_hits = trace_particles_boozer(
        field,
        points_filtered,
        vpar_init,
        tmax=tmax,
        mass=mass,
        charge=charge,
        comm=comm_world,
        Ekin=Ekin,
        stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        forget_exact_path=True,
        abstol=abstol,
        reltol=reltol,
        dt_save=1e-7
    )
    # print('done tracing')
    # time2 = time.time()
    # proc0_print("Elapsed time for tracing = ", time2 - time1)

    from firm3d.field.trajectory_helpers import compute_loss_fraction
    times, loss_frac = compute_loss_fraction(res_tys, tmin=1e-5, tmax=1e-2)
    out_dict = {
        'equilibria': boozmn_filename,
        'times': times,
        'loss_frac': loss_frac
    }
    np.save('perpitch_outputs/lossfrac_'+str(Bc),out_dict['loss_frac'][-1])

    # res_tys := [t,s,theta,zeta,v_par]
    # s is for res_tys[1]
    # arr = np.array(res_tys,dtype=object)
    # s_arr = []
    # drho = 0
    # for i in range(0,nParticles_filtered):
    #     s_arr.append(arr[i][:,1])
    #     drho += delta_rho(s_arr[i])
    # np.save('perpitch_outputs/drho'+str(Bc),drho)