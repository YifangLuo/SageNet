import numpy as np
import math
import pandas as pd
from scipy import interpolate
from scipy.integrate import quad
from scipy.integrate import solve_ivp


def int_FD(y):
    """
    Evaluate energy density and pressure for a massive neutrino in Fermi-Dirac dist. 
    x = pc/(kB*T), y = mc^2/(kB*T) = mc^2/(kB*T0)*a
    
    """

    I_rho = quad(integrand_rho, 0, 30, args=(y))
    I_p = quad(integrand_p, 0, 30, args=(y))

    coeff_massless = 7 * math.pi ** 4 / 120

    rho = I_rho[0] / coeff_massless  # ratio to that of a massless fermion
    p = I_p[0] / coeff_massless

    return np.array((rho, p))


def integrand_rho(x, y):
    return x ** 2 * math.sqrt(x ** 2 + y ** 2) / (math.exp(x) + 1)


def integrand_p(x, y):
    return x ** 4 / (3 * math.sqrt(x ** 2 + y ** 2) * (math.exp(x) + 1))

def solve_SGWB(Nv, Sv, j0, z0):
    N_span = (Nv[j0], Nv[-1]);
    N = Nv[j0:]
    ini_state = [z0, 0, math.exp(z0)]
    param = (interpolate.InterpolatedUnivariateSpline(Nv, Sv),)
    subhorizon.terminal = True;
    subhorizon.direction = 1
    # 意义：不同引力波的波长，对于每一个频率，j0是可以开始计算的时间
    # 对于每一个频率，其t_max/t_min是不固定的，而t_max - t_min是
    result = solve_ivp(tensor, N_span, ini_state,
                       method='LSODA',
                       # t_eval=N,
                       dense_output=True,
                       events=[subhorizon, ],
                       args=param,
                       rtol=1e-6, atol=[1e-10, 1e-20, 1e-20],
                       jac=jacobian,
                       )
    return result


def tensor(N, state, spline):
    """
    Dynamical system for tensor modes:
    
    z = ln(2*pi*f/aH)
    x = \dot T_h / H
    y = (2*pi*f/aH) * T_h
    sigma = -2 \dot H / 3H^2

    z' = 1.5*sigma - 1                
    x' = -3*x + 1.5*sigma*x - exp(z)*y
    y' = -y + 1.5*sigma*y + exp(z)*x 
    
    """

    z, x, y = state
    sigma = float(spline(N))

    dz = 1.5 * sigma - 1
    dx = -3 * x + 1.5 * sigma * x - math.exp(z) * y
    dy = -y + 1.5 * sigma * y + math.exp(z) * x

    return [dz, dx, dy]


def jacobian(N, state, spline):
    z, x, y = state
    sigma = float(spline(N))

    jac = np.array(
        (
            (0, 0, 0),
            (-math.exp(z) * y, -3 + 1.5 * sigma, -math.exp(z)),
            (math.exp(z) * x, math.exp(z), -1 + 1.5 * sigma),
        ),
        dtype=np.float64,
    )

    return jac


def subhorizon(N, state, spline):  # Deep-inside-the-Hubble threshold: k/aH = exp(5) ～ 150
    return state[0] - 5
