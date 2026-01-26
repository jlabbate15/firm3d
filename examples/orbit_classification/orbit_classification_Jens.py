import numpy as np

class OrbitClassification:

    def __init__(self, field, Ekin, mass, charge, helicity_M, helicity_N, barely_trapped_crit=2*np.pi*1.25, ripple_trapped_crit=0.5):
        r"""
        Initialize the OrbitClassification class.
        
        Args:
            field : BoozerMagneticField object.
            Ekin : Kinetic energy of the particle.
            mass : Mass of the particle.
            charge : Charge of the particle.
            helicity_M : Helicity M of the magnetic field strength.
            helicity_N : Helicity N of the magnetic field strength.
            barely_trapped_crit : Critical angle for barely trapped particle classification.
            ripple_trapped_crit : Critical value of dchi/dchi_predicted for ripple trapped particle classification.
        """
        self.field = field
        self.Ekin = Ekin
        self.mass = mass
        self.charge = charge
        self.helicity_M = helicity_M
        self.helicity_N = helicity_N
        self.nfp = field.nfp
        self.barely_trapped_crit = barely_trapped_crit # Critical angle for barely trapped particle classification
        self.ripple_trapped_crit = ripple_trapped_crit # Critical value of dchi/dchi_predicted for ripple trapped particle classification

        # If modB contours close poloidally, then use theta as mapping coordinate
        if helicity_M == 0:
            self.helicity_Mp = 1
            self.helicity_Np = 0
        # Otherwise, use zeta as mapping coordinate
        else:
            self.helicity_Mp = 0
            self.helicity_Np = self.nfp

    def chi_eta_to_theta_zeta(self,chi, eta):
        r"""
        Convert helical angles (chi, eta) to (theta, zeta).

        Args:
            chi : Helical angle chi.
            eta : Mapping angle eta.
        Returns:
            theta : Poloidal angle.
            zeta : Toroidal angle.
        """
        denom = self.helicity_Np * self.helicity_M - self.helicity_N * self.helicity_Mp
        theta = (self.helicity_Np * chi - self.helicity_N * eta) / denom
        zeta = (self.helicity_Mp * chi - self.helicity_M * eta) / denom

        return theta, zeta
    
    def classify_orbit(self, res_ty, res_hit):
        r"""
        Classify the orbit of a particle based on its trajectory and mirror points.

        Args:
            res_ty : Array containing the trajectory of the particle.
            res_hit : Array containing the hit points of the particle.

        """

        # Compute all of the times when the particle bounces off the vpar plane
        bounce_times = []
        nhits = len(res_hit[:,0])
        for j in range(nhits):
            if (res_hit[j,1]==0): # vpar plane was hit
                bounce_times.append(res_hit[j,0])

        nbounce = len(bounce_times)

        # In case axis = 1, 2 is used, we need to unwrap the angle 
        thetas = np.unwrap(res_ty[:,2])

        point = np.zeros((1, 3))
        point[0,:] = res_ty[0,1:4]  # Initial position of the particle
        vpar_init = res_ty[0,4]  # Initial parallel velocity of the

        # Compute trapping parameter lam = vperp^2/(v^2B)
        self.field.set_points(point)
        modB_0 = self.field.modB()[0,0]
        lam = (2*self.Ekin/self.mass - vpar_init**2)/(modB_0*2*self.Ekin/self.mass)
        modB_crit = 1/lam

        Jpars = []
        s_means = []
        dchis = [] 
        dchis_predicted = []
        gammacs = []
        dss = []
        dalphas = []
        # Iterate over bounce segments 
        for j in range(nbounce-1):
            index_start = np.argmin(np.abs(bounce_times[j] - res_ty[:,0]))
            index_end = np.argmin(np.abs(bounce_times[j+1] - res_ty[:,0]))
            ds = res_ty[index_end,1] - res_ty[index_start,1]
            dss.append(ds)

            dtheta = thetas[index_end] - thetas[index_start]
            dzeta = res_ty[index_end,3] - res_ty[index_start,3]
            dchi = self.helicity_M*dtheta - self.helicity_N*dzeta
            dchis.append(np.abs(dchi))

            mean_s = np.mean(res_ty[index_start:index_end+1,1])

            # Find approximate mirror points on this s surface 
            chi_grid = np.linspace(0,2*np.pi,100)
            theta, zeta = self.chi_eta_to_theta_zeta(chi_grid, np.zeros_like(chi_grid))
            points = np.zeros((len(chi_grid.flatten()),3))
            points[:,0] = mean_s
            points[:,1] = theta
            points[:,2] = zeta
            self.field.set_points(points)
            iota_s = self.field.iota()[0,0]

            dalpha = dtheta - iota_s*dzeta
            dalphas.append(dalpha)
            gammac = (2/np.pi)*np.arctan(np.abs(ds)/np.abs(dalpha))
            gammacs.append(gammac)

            modB = self.field.modB()[:,0]
            mirror_loc = np.argmin(np.abs(modB - modB_crit))
            chi_mirror = chi_grid[mirror_loc]
            min_loc = np.argmin(modB)
            chi_min = chi_grid[min_loc]
            dchi_predicted = np.min([np.abs(2*(chi_mirror - chi_min)), np.abs(2*(chi_mirror - (chi_min+2*np.pi))), np.abs(2*(chi_mirror - (chi_min-2*np.pi)))])
            dchis_predicted.append(dchi_predicted)

            # Compute Jpar along trajectory 
            points = np.zeros((index_end-index_start+1,3))
            points[:,0] = res_ty[index_start:index_end+1,1]
            points[:,1] = res_ty[index_start:index_end+1,2]
            points[:,2] = res_ty[index_start:index_end+1,3]
            self.field.set_points(points)
            bdotgradzeta = self.field.modB()[:,0]/(self.field.G()[:,0] + self.field.iota()[:,0]*self.field.I()[:,0])
            vpar = res_ty[index_start:index_end+1,4]

            # Compute integral using trapezoid rule 
            vpar_center = 0.5*(vpar[1::]+vpar[0:-1])
            bdotgradzeta_center = 0.5*(bdotgradzeta[1::]+bdotgradzeta[0:-1])
            delta_zeta = points[1::,2]-points[0:-1,2]
            Jpar = np.sum(vpar_center * delta_zeta / bdotgradzeta_center)
            Jpars.append(Jpar)

        # John's add: Find drho:
        def delta_rho(s): # returns the total ds for one particle
            # s = np.array(poinc.s_all)
            s_max = np.max(s)
            s_min = np.min(s)
            drho = s_max**(0.5)-s_min**(0.5) # one for each s,eta initialization
            return np.sum(drho)
        drho = delta_rho(res_ty[:,1])

        dchis = np.array(dchis)
        dchis_predicted = np.array(dchis_predicted)

        # Now classify the trapping state of the particle
        if nbounce < 2:
            # Particle never mirrored
            status = np.ones_like(dchis)*-1
            banana_frac = 0.0
            barely_trapped_frac = 0.0
            ripple_trapped_frac = 0.0
            Jpar_var = 0.0
            gammac_mean = 0.0
            ntransitions = 0
        else:

            # Classify the trapping state of the particle
            # 0: banana trapped, 1: barely trapped, 2: ripple trapped,
            status = np.zeros_like(dchis)
            status[dchis > self.barely_trapped_crit] = 1
            status[dchis < self.ripple_trapped_crit * dchis_predicted] = 2

            barely_trapped_frac = np.count_nonzero(dchis > self.barely_trapped_crit) / len(dchis)
            ripple_trapped_frac = np.count_nonzero(dchis < self.ripple_trapped_crit * dchis_predicted) / len(dchis)
            banana_frac = np.count_nonzero((dchis <= self.barely_trapped_crit) * (dchis >= self.ripple_trapped_crit)) / len(dchis)

            # Check for transitions between trapping states
            ntransitions = np.count_nonzero(status[0:-1] != status[1::]) 

            if nbounce > 1: 
                gammac_mean = np.mean(gammacs)
            else:
                gammac_mean = 0.0

            if nbounce > 3: 
                # Look at full bounce period 
                Jpar_full = Jpars[0:-1] + Jpars[1::]
                # Compute normalized variation in Jpar
                Jpar_var = np.std(Jpar_full)/np.mean(Jpar_full)
            else:
                Jpar_var = 0.0

        particle_dict = {
            'losttime': res_ty[-1,0], # Total time the particle was lost
            'nbounce': nbounce, # Number of bounce segments
            'bounce_times': bounce_times, # List of times when bounces occur (len = nbounce)
            'lam': lam, # Trapping parameter, 1/modBcrit = vperp^2/(v^2B)
            'point0': point, # Initial position of the particle
            'vpar0': vpar_init, # Initial parallel velocity of the particle

            # All of these quantities have length nbounce-1
            'status': status, # Array of trapping states for each bounce segment (0: banana trapped, 1: barely trapped, 2: ripple trapped)
            'dss': dss, # List of change in s values over each bounce segment
            'dalphas': dalphas, # List of change in alpha values over each bounce segment
            'dchis': dchis, # List of change in chi values over each bounce segment
            'dchis_predicted': dchis_predicted, # List of predicted change in chi values over each bounce segment based on helicity
            'gammacs': gammacs, # List of gamma_c values for each bounce segment
            'Jpars': Jpars, # List of Jpar values for each bounce segment
            'gammacs': gammacs, # List of gamma_c values for each bounce segment
            's_means': s_means, # List of mean s values for each bounce segment

            # Cumulative statistics
            'banana_frac': banana_frac, # Fraction of time spent in banana trapped state
            'barely_trapped_frac': barely_trapped_frac, # Fraction of time spent in barely trapped state
            'ripple_trapped_frac': ripple_trapped_frac, # Fraction of time spent in ripple trapping state
            'ntransitions': ntransitions, # Number of transitions between trapping states
            'Jpar_var': Jpar_var, # Normalized variation in Jpar over full bounce periods
            'gammac_mean': gammac_mean, # Mean value of gamma_c over all bounce segments

            # John's add
            'drho': drho,
        }
        return particle_dict