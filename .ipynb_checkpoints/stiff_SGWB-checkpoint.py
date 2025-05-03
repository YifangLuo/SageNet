# This is a file module which contains classes and functions 
# which calculate the cosmological model of LCDM + stiff + primordial SGWB

import json
import os, yaml, math
import time
import multiprocessing as mp
import numpy as np
from numpy import concatenate as cat
from scipy import interpolate, integrate

from global_param import *
from functions import int_FD, solve_SGWB
from LCDM_stiff_Neff import LCDM_SN

class LCDM_SG(LCDM_SN):
    """
    Cosmological model: LCDM + stiff component + constant Delta N_eff

    The free/base parameters of the model are: 
    'Omega_bh2', 'Omega_ch2', 'H0', 'DN_eff', 'A_s', 'r', 'n_t', 'cr', 'T_re', 'DN_re', 'kappa10'.
    - [H0] = km s^-1 Mpc^-1, [T_re] = GeV.
    - Set cr > 0 if the consistency relation is assumed, otherwise set cr <= 0 and provide (n_t, DN_re).
    - DN_re is the number of e-folds from the end of inflation to the end of reheating, 
      assuming a^{-3} matter-like evolution.
    - kappa10 := rho_stiff/rho_photon at 10 MeV.

    There are three ways to instantiate a model with desired base parameters: 
    1. input a yaml file,  2. input a dictionary, 
    3. specify the parameters by keyword arguments.
    
    They will be stored in the 'obj_name.cosmo_param' attribute as a dictionary
    and can be modified later on.
    
    Once a successful 'SGWB_iter' calculation has been performed for an instance, 
    if for some reason you would like to reuse this instance with a new set of 
    parameters, you MUST run 'obj_name.reset()' first to reset the status, and 
    then modify the 'obj_name.cosmo_param' dictionary with desired values.
    
    """
    def __init__(self, *args,
                 **kwargs):
        LCDM_SN.__init__(self, *args, **kwargs)
        self.reset()


    def reset(self):
        #LCDM_SN.derived_param(self)
        if hasattr(self, 'DN_eff_orig') and (self.DN_eff_orig is not None):
            self.cosmo_param['DN_eff'] = self.DN_eff_orig
        self.DN_eff_orig = None
        self.SGWB_converge = False     # Whether the SGWB has been successfully computed


    #@property


    def run_SGWB_single(self, freq):
        """
        Integrate the equation of motion of primordial tensor fluctuations 
        for a single frequency and given expansion history
        """

        P_t = self.Tensor_power(freq)            # Primordial tensor power spectrum

        j = self.find_index_hc(freq+3)           # solver begins at kc/aH <= 10^{-3}
        z0 = (freq-self.f_hor[j]) * ln10         # convert to ln(kc/aH)
        print('Running a SGWB calculation.')
        result = solve_SGWB(self.Nv, self.sigma, j, z0)
        N_this = self.Nv[(self.Nv >= result.sol.t_min) & (self.Nv <= result.sol.t_max)]
        [zf_this, xf_this, yf_this] = result.sol(N_this)

        Th_this = np.divide(yf_this, np.exp(zf_this))
        Oj_this = np.multiply(xf_this, Th_this)/3 * P_t
        Ogw_this = (np.power(xf_this,2) + np.power(yf_this,2))/24 * P_t + Oj_this
        Opgw_this = (-5*np.power(xf_this,2) + 7*np.power(yf_this,2))/72 * P_t

        # high-frequency, post-horizon-reentry regime
        if (N_this[-1]<self.Nv[-1]):
            N_hf = self.Nv[self.Nv > N_this[-1]]
            f_hf = self.f_hor[self.Nv > N_this[-1]]

            coeff = math.sqrt(0.5*(xf_this[-1]**2+yf_this[-1]**2))
            Th_hf = coeff * np.exp(-zf_this[-1] + N_this[-1] - N_hf)      # the rms T_h, time-averaged
            xf_hf = np.multiply(Th_hf, np.power(10, freq-f_hf))           # the rms x_f, time-averaged
            Oj_hf = -np.power(Th_hf,2)/3 * P_t
            Opgw_hf = np.power(xf_hf,2)/36 * P_t
            Ogw_hf = Opgw_hf * 3 + Oj_hf

            N_this = cat((N_this, N_hf), axis=None)
            Th_this = cat((Th_this, Th_hf), axis=None)
            Oj_this = cat((Oj_this, Oj_hf), axis=None)
            Ogw_this = cat((Ogw_this, Ogw_hf), axis=None)
            Opgw_this = cat((Opgw_this, Opgw_hf), axis=None)

        return [N_this, Th_this, Oj_this, Ogw_this, Opgw_this]


    def run_SGWB(self):
        """
        Solve for selected frequencies and given expansion history
        using multiprocessing parallelism
        """
        print('Starting a SGWB calculation.')
        with mp.Pool(processes=4) as pool:
            res_it = pool.imap(self.run_SGWB_single, self.f, chunksize=3)
            res = [y for y in res_it]

        # Each element in the following lists is for a frequency channel
        self.N_hc = [single_sol[0] for single_sol in res]  # Number of e-folds, starting at the horizon crossing for each frequency
        self.Th = [single_sol[1] for single_sol in res]    # Tensor transfer function
        self.Oj = [single_sol[2] for single_sol in res]    # \dot T_h * T_h / 3H * P_t
        self.Ogw = [single_sol[3] for single_sol in res]   # dOmega_GW / dlnf
        self.Opgw = [single_sol[4] for single_sol in res]  # dOmega_pGW / dlnf


    def SGWB_iter(self):
        """
        Main numerical scheme: 
        Iteration method that yields self-consistent cosmology including the stiff-amplified primordial SGWB, 
        for which the extra radiation due to the SGWB is mimicked by a constant Delta N_eff.
        
        """
        # Exclude some corner cases
        if self.cosmo_param['r'] <= 0:
            print('Must set a positive r to calculate the inflationary GWs!')
            return

        if self.derived_param['N_inf'] is None:
            print('High-end cutoff frequency has not been set properly.')
            return

        # Main calculation starts here!
        print(f'Main calculation started! Now it is {"" if self.SGWB_converge else "not"} converged')
        if self.SGWB_converge == False:
            Omega_nu = Omega_nh2/self.derived_param['h']**2

            self.DN_eff_orig = self.cosmo_param['DN_eff']
            # Record the original input value of DN_eff before entering the iterations

            self.DN_gw = [0]; DN_gw_new = 0
            DN_gw_min = 0; DN_gw_max = 10
            while True:    # main iteration
                print('Starting main iteration.')
                self.gen_expansion()
                self.construct_f()
                self.run_SGWB()
                self.int_SGWB()

                DN_gw_new = Neff0 * self.g2[-1] / Omega_nu

                #print(DN_gw_new, self.DN_gw[-1], DN_gw_min, DN_gw_max, 7/8*(4/11)**(4/3)*self.cosmo_param['DN_eff']/self.derived_param['rho_re'])
                if self.DN_eff_orig + DN_gw_new > 5:
                    #print('Total N_eff too large! Shorten the stiff era or lower the blue tilt n_t.')
                    self.cosmo_param['DN_eff'] = self.DN_eff_orig
                    self.DN_eff_orig = None
                    return

                # Break when the required convergence precision is met!
                if abs((Neff0+self.DN_eff_orig+DN_gw_new)/(Neff0+self.DN_eff_orig+self.DN_gw[-1]) - 1) < 1e-4:
                    self.cosmo_param['DN_eff'] = self.DN_eff_orig + DN_gw_new
                    break

                # Use bisection method to find the next point to shoot
                if DN_gw_new > self.DN_gw[-1] > DN_gw_min and DN_gw_max >= self.DN_gw[-1]:
                    DN_gw_min = self.DN_gw[-1]
                elif DN_gw_new < self.DN_gw[-1] < DN_gw_max and DN_gw_min <= self.DN_gw[-1]:
                    DN_gw_max = self.DN_gw[-1]

                if 0 < DN_gw_min <= DN_gw_max < 10:
                    DN_gw_new = (DN_gw_min + DN_gw_max)/2

                self.cosmo_param['DN_eff'] = self.DN_eff_orig + DN_gw_new
                self.DN_gw.append(DN_gw_new)
            # End of main iteration


            #print(DN_gw_new, self.DN_gw, self.cosmo_param['DN_eff'])
            self.SGWB_converge = True
            self.hubble = math.log10(2*math.pi) + self.f_hor + (self.Nv[-1]-self.Nv)/ln10       # log10(H/s^-1), H = 2pi * f_hor / a
            self.DN_gw = Neff0 * np.multiply(self.g2, np.exp(2*(self.f_hor-self.f_hor[-1])*ln10+2*(self.Nv-self.Nv[-1]))) / Omega_nu
            # Obtain the entire evolution of the DN_eff due to SGWB. It is actually rho_GW(N) / (rho_{gamma,0}*7/8*(4/11)**(4/3)).
            # Now self.DN_gw[-1] + self.DN_eff_orig = self.cosmo_param['DN_eff'].

            self.Ogw_today = np.array([self.Ogw[i][-1] for i in range(len(self.f))])
            self.Opgw_today = np.array([self.Opgw[i][-1] for i in range(len(self.f))])
            self.Oj_today = np.array([self.Oj[i][-1] for i in range(len(self.f))])
            self.log10OmegaGW = np.log10(self.Ogw_today - self.Oj_today)  # log10(Omega_GW(f))
            # Ignoring the negative super-horizon contribution from Omega_j, for the moment...

            # Interpolate on uniform-spaced frequency bins and output
            self.f_grid = np.arange(-18.5,6.5,.25)
            self.log10OmegaGW_grid = -40 * np.ones_like(self.f_grid)
            Ogw_spl = interpolate.CubicSpline(np.flip(self.f), np.flip(self.log10OmegaGW))
            cond = self.f<self.f[0]-.1
            Ogw_lin = interpolate.CubicSpline(np.flip(self.f[cond]), np.flip(self.log10OmegaGW[cond]), bc_type='natural')
            add_boundary_knots(Ogw_lin)
            self.log10OmegaGW_grid[self.f_grid<=self.f[0]] = Ogw_spl(self.f_grid[self.f_grid<=self.f[0]])
            self.log10OmegaGW_grid[self.f_grid>self.f[0]] = Ogw_lin(self.f_grid[self.f_grid>self.f[0]], nu=0)


            ###  Extra radiation (e.g., SGWB) parameterized as kappa_rad(T_i) for AlterBBN

            self.kappa_r = self.cosmo_param['DN_eff']* 7/8*(4/11)**(4/3) * z_ratio**4
            # Using the final asymptotic value of Delta N_eff, since for all reasonable T_re (>~ 1 MeV), 
            # Delta N_eff,GW has already (or almost) reached its asymptotic value by T_i=27e9 K for AlterBBN.

    # End of SGWB_iter



    def Tensor_power(self, freq):
        return self.derived_param['A_t'] * np.power((10**freq)/f_piv, self.derived_param['nt'])


    def find_index_hc(self, freq):
        """
        Find the index in the number of e-folds array at which 
        the horizon size is closest to freq, i.e., freq ~ aH/2pi
        """
        j = 0
        while freq <= self.f_hor[j] and j<len(self.N)-1:
            j = j+1
        if (j >= 1): j = j-1

        return j


    def construct_f(self):
        """
        Construct the array of sampled frequencies to calculate tensor transfer functions,
        chosen empirically -- more points around transition!
        """
        fmax = self.f_hor[0]; fmin = min(self.f_hor); fcmb = math.log10(f_piv)
        if self.derived_param['nt']>0:            # Only count modes whose superhorizon power is less than unity
            fmax = min(fmax, (-math.log10(self.derived_param['A_t']))/self.derived_param['nt']+math.log10(f_piv))

        f = fmax+np.zeros(1); f = cat((f, f[-1]-np.arange(1,25)*1e-3), axis=None); f = cat((f, f[-1]-np.arange(1,14)*2e-3), axis=None)
        f = cat((f, f[-1]+5e-2-np.logspace(-1.28,-1, num=17)), axis=None); f = cat((f, f[-1]+1e-1-np.logspace(-1,0, num=38)[1:]), axis=None)
        f = cat((f, f[-1]-np.arange(1,11)*.2), axis=None)

        if fmax >= self.f_re:
            f = f[np.logical_or(f>=self.f_re+.2, f>=fmax)]; f = f[f>=fmax-.4]
            f = cat((f, np.arange(f[-1]-.5, self.f_re+1.5, -.5)), axis=None)          # before T_re, during reheating
            f = cat((f, np.arange(f[-1]-.1, self.f_re+.8, -.1)), axis=None);          # approaching f_re
            f = cat((f, np.arange(f[-1]-.05, self.f_re+.4, -.05)), axis=None)
            f = cat((f, np.arange(f[-1]-.01, self.f_re-.22, -.01)), axis=None);        # through f_re
            f = cat((f, np.arange(f[-1]-.022, self.f_re-.5, -.02)), axis=None)
            f = cat((f, np.arange(f[-1]-.05, self.f_re-1, -.05)), axis=None);         # away from f_re
            f = cat((f, np.arange(f[-1]-.2, self.f_re-3, -.2)), axis=None)

        if self.rhostiff_re > self.rhorad_re:
            f = cat((f, np.arange(f[-1]-1, self.f_re-math.log10(self.rhostiff_re/self.rhorad_re)+1, -1)), axis=None)     # after T_re, before T_sr
            f = cat((f, np.arange(f[-1]-.2, self.f_re-math.log10(self.rhostiff_re/self.rhorad_re)-2, -.2)), axis=None)   # through T_sr (the ankle)

        f = f[f>=fcmb]
        f = cat((f, np.arange(f[-1]-1, fcmb, -1)), axis=None)                         # RD, before z_eq
        f = cat((f, np.arange(f[-1]-.2, fmin-1, -.2)), axis=None); f = f[f>=fmin]     # through z_eq to the present

        self.f = f


    def int_SGWB(self):
        """
        Integrate the SGWB spectrum to obtain the bolometric energy fractions at each moment.
        This step is important since it calculates the Delta N_eff due to the primordial SGWB.
        """
        self.g2 = np.zeros_like(self.Nv)      # total Omega_gw
        self.w2 = np.zeros_like(self.Nv)      # total Omega_pgw

        # Patch SGWB solutions into matrices for integration over f
        M_N_hc = -np.ones((len(self.f),len(self.Nv)))
        M_Ogw = np.zeros_like(M_N_hc); M_Opgw = np.zeros_like(M_N_hc); M_Oj = np.zeros_like(M_N_hc)
        for i in range(len(self.f)):
            #cond = self.N_hc[i]>=self.Nv[self.find_index_hc(self.f[i]-3)]
            # Only count GW energy density when each mode is at least 3 decades inside the horizon
            cond = self.N_hc[i]>=self.N_hc[i][0]
            M_N_hc[i, (mask := np.isin(self.Nv, self.N_hc[i][cond]))] = self.N_hc[i][cond]
            M_Ogw[i,mask] = self.Ogw[i][cond]; M_Opgw[i,mask] = self.Opgw[i][cond]; M_Oj[i,mask] = self.Oj[i][cond]

        M_N_hc = np.transpose(M_N_hc); M_Ogw = np.transpose(M_Ogw); M_Opgw = np.transpose(M_Opgw); M_Oj = np.transpose(M_Oj)

        # Do the integration!
        for i in range(len(self.Nv)):
            #ind_int = (M_N_hc[i]>=0) & (self.f<=self.f[0]-3)    # indices of modes whose frequencies are at least 3 decades less than f_inf
            ind_int = (M_N_hc[i]>=0)
            f_int = np.flip(self.f[ind_int])                           # log10(f/Hz), increasing order
            Ogw_int = np.flip(M_Ogw[i,ind_int]); Opgw_int = np.flip(M_Opgw[i,ind_int]); Oj_int = np.flip(M_Oj[i,ind_int])

            if len(f_int)>=2:
                # do not consider the negative super-horizon contribution from Omega_j, for the moment...    
                self.g2[i] = integrate.simpson(Ogw_int-Oj_int, f_int) * ln10
                #self.g2[i] = integrate.simpson(Ogw_int, f_int) * ln10
                self.w2[i] = integrate.simpson(Opgw_int, f_int) * ln10
            else:
                self.g2[i] = 0; self.w2[i] = 0

            #fx = np.arange(f_int.min(), f_int.max(), .01)
            #spline_gw = interpolate.InterpolatedUnivariateSpline(f_int, Ogw_int-Oj_int)
            #spline_pgw = interpolate.InterpolatedUnivariateSpline(f_int, Opgw_int)   # Opgw can be negative
            #self.g2[i] = integrate.simpson(spline_gw(fx), fx) * math.log(10)
            #self.w2[i] = integrate.simpson(spline_pgw(fx), fx) * math.log(10)


def add_boundary_knots(spline):
    """
    Add knots infinitesimally to the left and right.

    Additional intervals are added to have zero 2nd and 3rd derivatives,
    and to maintain the first derivative from whatever boundary condition
    was selected. The spline is modified in place.
    """
    # determine the slope at the left edge
    leftx = spline.x[0]
    lefty = spline(leftx)
    leftslope = spline(leftx, nu=1)

    # add a new breakpoint just to the left and use the
    # known slope to construct the PPoly coefficients.
    leftxnext = np.nextafter(leftx, leftx - 1)
    leftynext = lefty + leftslope*(leftxnext - leftx)
    leftcoeffs = np.array([0, 0, leftslope, leftynext])
    spline.extend(leftcoeffs[..., None], np.r_[leftxnext])

    # repeat with additional knots to the right
    rightx = spline.x[-1]
    righty = spline(rightx)
    rightslope = spline(rightx,nu=1)
    rightxnext = np.nextafter(rightx, rightx + 1)
    rightynext = righty + rightslope * (rightxnext - rightx)
    rightcoeffs = np.array([0, 0, rightslope, rightynext])
    spline.extend(rightcoeffs[..., None], np.r_[rightxnext])