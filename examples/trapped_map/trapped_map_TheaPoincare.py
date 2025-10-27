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

import matplotlib.pyplot as plt


# COMMON USER INPUTS #
boozmn_filename = "../inputs/boozmn_equil_G1600_DESC_fixed.nc"
Ekins = FUSION_ALPHA_PARTICLE_ENERGY*np.array([1,0.001]) # left->right order will be left->right order on plot
neta_poinc = 12  # Number of eta initial conditions for poincare
ns_poinc = 60  # Number of s initial conditions for poincare
Nmaps = 1500  # Number of Poincare return maps to compute
modBin = 5.95
tmax = 1e-2
#######################

fig, ax = plt.subplots(nrows=1, ncols=3) # setup for plotting

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


# Setup logging to redirect output to file
# setup_logging(f"stdout_trapped_map_{resolution}_{comm_size}.txt")

# print("start BRI")

# First find frequency profiles in perfect case
print("Start frequency profiles")
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

# print("start tracing")
omega_zetas = [0,0]
s_profs = [0,0]
i=0
for Ekin in Ekins:
    time1 = time.time()
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
    proc0_print("poincare tracing time: ", time2 - time1)

    # Compute frequencies
    omega_eta_prof, omega_b_prof, s_profs[i] = poinc.compute_frequencies()
    omega_zetas[i] = omega_eta_prof/omega_b_prof

    i+=1


# Second find Poincare plots in non-perfect case
print("Start Poincare plots")
# bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world, helicity_N=helicity_N, helicity_M=helicity_M) # specify helicities to filter QS-breaking modes
bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm_world)

# print("start IBF")

field = InterpolatedBoozerField(
    bri,
    degree,
    ns_interp=ns_interp,
    ntheta_interp=ntheta_interp,
    nzeta_interp=nzeta_interp,
    stellsym=True
)

# print("start tracing")
i=0
for Ekin in Ekins:
    time1 = time.time()
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
    proc0_print("poincare tracing time: ", time2 - time1)

    ax = poinc.plot_poincare(ax=ax,j=i,filename='poinc_'+str(i),save_points=True)
    
    i+=1


# Rational numbers
p_max=4
q_max=4
res_range_min = -2
res_range_max = 2
res_arr = np.array([])
# q_arr = np.array([])
res_arr_set = 0

# including the zero resonance
res_arr = np.append(res_arr,0)

for p in range(1,p_max+1): # include the zero resonance
    for q in range(1,q_max+1):
            condition1 = p/q >= res_range_min and p/q <= res_range_max
            if ~np.isin(p/q, res_arr) and condition1:
                res_arr = np.append(res_arr,p/q)
                res_arr = np.append(res_arr,-p/q)


# Plotting #

# Omega Zeta Plot (figure c)
ax[2].plot(omega_zetas[0],s_profs[0]**0.5,'b',label=r"KE = $3.5$ MeV")
ax[2].plot(omega_zetas[1],s_profs[1]**0.5,'r',label=r"KE = $0.001 * 3.5$ MeV")
ax[2].set_xlabel(r'$\omega_{\zeta}$')
ax[2].yaxis.set_label_position("right")
ax[2].yaxis.tick_right()
ax[2].set_ylabel(r'$\rho$')
ax[2].set_ylim(0.0,1.0)
ax[2].set_xlim(-2.1,0.1)
ax[2].legend(loc='upper left',bbox_to_anchor=(-1.46, 1.15))
ax[0].set_xlim([0,np.pi])
ax[1].set_xlim([0,np.pi])
ax[0].set_xticks([0, 1, 2, 3])
ax[0].set_xticklabels([str(0), str(1), str(2), str(3)])
ax[1].set_xticks([0, 1, 2, 3])
ax[1].set_xticklabels([str(0), str(1), str(2), str(3)])
colors_plot = ['bx','rx']
for res in res_arr:
    ax[2].plot([res,res],[0,1],'k--',lw=0.5)
    i=0
    for omega_zeta in omega_zetas: # loop through each energy
        for j in range(0,len(s_profs[i])-1): # loop through each s
            res_dist_curr = omega_zeta[j] - res
            res_dist_next = omega_zeta[j+1] - res
            if np.sign(res_dist_curr)==np.sign(res_dist_next):
                continue # no res crossing found
            else: # res crossing found
                xs=np.linspace(omega_zeta[j],omega_zeta[j+1],100)
                ses = np.linspace(s_profs[i][j]**0.5,s_profs[i][j+1]**0.5,100)
                x_plot=999999999
                k_plot=0
                k=0
                for x in xs:
                    if abs(x-res) < abs(x_plot-res):
                        x_plot = x
                        k_plot = k
                    k+=1
                ax[2].plot(x_plot,ses[k_plot],colors_plot[i])
        i+=1

# Extra for Poincare Plot 1
ax[1].yaxis.set_visible(False)
plt.savefig("poincare_omega_Thea.pdf")

np.save("freq",np.array(omega_zetas,dtype=object))
np.save("s",np.array(s_profs,dtype=object))