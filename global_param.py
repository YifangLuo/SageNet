import numpy as np
import os, sys, math
from scipy import interpolate
from astropy.cosmology import Planck18 as cosmo
from astropy import constants as const
import astropy.units as u
from functions import int_FD

#######   Constants. Do not change.   #######

# Base

kB = const.k_B.to(u.eV/u.K).value          # eV K^-1
hbar = const.hbar.to(u.eV*u.s).value       # eV s
c = const.c.to(u.cm/u.s).value             # cm s^-1
eV = (u.eV/(const.c**2)).to(u.g).value     # g*c^2
GN = const.G.value*1e3                     # cm^3 g^-1 s^-2
parsec = u.parsec.to(u.cm)                 # cm
yr = u.yr.to(u.s)                          # seconds in a Julian year
ln10 = math.log(10)

# Derived

rc = math.pi**2/(15*(hbar*c)**3)           # eV^-3 cm^-3, radiation constant = pi^2/(15*hbar^3*c^3)
m_Pl = math.sqrt(hbar*c**3/(GN*eV))*1e-9   # GeV/c^2, Planck mass = sqrt(hbar*c/G)
M_Pl = m_Pl/math.sqrt(8*math.pi)           # GeV/c^2, reduced Planck mass
f_piv = 0.05 *c/(2*math.pi*1e6*parsec)     # s^-1, CMB pivot scale = 0.05 Mpc^-1


#######   Cosmological Parameters   #######

# Base

TCMB = cosmo.Tcmb0.value  # K
TCMB_GeV = 1e-9*kB*TCMB   # GeV

m_nu = 60                 # meV, for a single massive neutrino eigenstate

Neff0 = 3.044
tau_n = 878.4             # s, neutron lifetime from PDG 2022

# Derived

Omega_ph2 = rc * (kB*TCMB)**4 * (8*math.pi*GN*eV/3) *(0.1*parsec)**2         # photons
Omega_nh2 = Omega_ph2 * 7/8 * (4/11)**(4/3) * Neff0                          # relativistic SM neutrinos
Omega_orh2 = Omega_ph2 + Omega_nh2                                           # SM radiation after e+e- annihilation

Tnu = TCMB * (4/11)**(1/3)                # effective thermal temperature shared by all neutrinos today
nu_today = m_nu*1e-3 / (kB*Tnu)
[rho_nu0, p_nu0] = int_FD(nu_today)
Omega_mnuh2 = Omega_nh2/3 * rho_nu0       # the massive neutrino eigenstate


######      Thermal history from 20 keV ~ 10^6 GeV     ######

thdata = np.loadtxt(os.path.dirname(__file__) + '/th.dat')
T_th = thdata[:,0]; T_max = T_th[-1]
N_th = thdata[:,2]; N_max = N_th[-1]; N_fin = N_th[0]
rho_th = thdata[:,5]; rhop_th = thdata[:,6]

spl_rho = interpolate.CubicSpline(N_th, rho_th)
spl_rhop = interpolate.CubicSpline(N_th, rhop_th)

spl_T_N = interpolate.CubicSpline(np.log10(T_th), N_th)
N_10 = spl_T_N(-2); a_10 = np.exp(-N_10)

#####       AlterBBN      #########

T_i = 27. * 8.617330637338339e-5   # Initial temperature at 27e9 K
N_i = spl_T_N(math.log10(T_i))
z_ratio = TCMB_GeV/T_i*np.exp(N_i)  # T_CMB/(T_i*a_i)


#####    Gravitational-wave data    #####

f_yr = 1/yr     # Hz, reference frequency used for PTA
f_LIGO = 25     # Hz, reference frequency used by LIGO-Virgo


#####     BBN observational data, obsolete    #####

Neff_A = (2.86 + 0.57)*Neff0/3      # 95% upper bound, BBN, Aver+ 2015 + Cooke+ 2014
Neff_I = 3.58 + 0.40                # 95% upper bound, BBN, Izotov+ 2014 + Cooke+ 2014

Neff_l = 2.99 + 0.43                # 95% upper bound, CMB+BAO+Y_P(Aver+15), low-z acoustic scales + Silk damping scale
Neff_lnp = 2.97 + 0.58              # same as above, CMB+BAO, without Y_P

T_np = 1.293              # MeV   neutron/proton ratio freeze-out
N_np = math.log(T_np*1e6/kB/TCMB)
T_D = 1/14                # MeV   Deuterium synthesis
N_D = math.log(T_D*1e6/kB/TCMB)
N_BBN = [18.2, 23.3]      # ~[0.02 3] MeV  to safely include the whole BBN process