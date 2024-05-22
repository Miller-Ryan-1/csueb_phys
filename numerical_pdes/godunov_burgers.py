import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from IPython.display import display, clear_output


def second_order_godunov_burgers(u0, dx, dt, t_max):
    """Solve the inviscid Burgers' equation using a second-order Godunov scheme with MUSCL"""
    nt = int(t_max / dt)
    nx = len(u0)
    u = np.zeros((nt + 1, nx))
    u[0, :] = u0

    for n in range(nt):
        u_n = u[n, :]
        uL = np.zeros(nx - 1)
        uR = np.zeros(nx - 1)
        flux = np.zeros(nx)

        for i in range(1, nx - 1):
            duL = minmod(u_n[i] - u_n[i-1], u_n[i+1] - u_n[i])
            duR = minmod(u_n[i+1] - u_n[i], u_n[i+2] - u_n[i+1] if i + 2 < nx else 0)

            uL[i] = u_n[i] + 0.5 * duL
            uR[i] = u_n[i+1] - 0.5 * duR

        for i in range(1, nx - 1):
            flux[i] = godunov_flux(uL[i], uR[i])

        for i in range(1, nx - 1):
            u[n + 1, i] = u_n[i] - dt / dx * (flux[i] - flux[i - 1])
        
        # Boundary conditions
        u[n + 1, 0] = u[n + 1, -1] = 0

    return u



def godunov_flux(ul, ur):
    '''
    Godunov's flux function
    '''
    if ul > ur:
        if ul > 0 and ur > 0:
            return 0.5 * ul**2
        elif ul < 0 and ur < 0:
            return 0.5 * ur**2
        else:
            return 0
    else:
        if ul + ur > 0:
            return 0.5 * ul**2
        else:
            return 0.5 * ur**2
        
def minmod(a, b):
    '''
    Minmod function for slope limiting - improves Gudanov's w/ MUSCL - Monotonic Upstream Centered Schemes for Conservation Laws
    '''
    if a * b <= 0:
        return 0
    else:
        return min(abs(a), abs(b)) * np.sign(a)
    

