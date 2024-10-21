import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from IPython.display import display, clear_output


# Godunov scheme implementation with time-dependent a and a_dot
def godunov_scheme(L, dr, k, c, alpha, t_max, u_0_x, u_t_0, u_t_L):

    r_grid, u0 = build_u0(L, dr, u_0_x, u_t_0, u_t_L)

    dt = compute_dt(u0, r_grid, dr, k, c, alpha, t_max)
    U = []
    U.append(u0)
    u = u0.copy()
    t = 0
    while t < t_max + dt:
        t += dt
        u_new = u.copy()

        for i in range(1, len(u) - 1):
            r_left = r_grid[i - 1]
            r_center = r_grid[i]
            r_right = r_grid[i + 1]

            F_i_plus_1_2 = hll_flux(u[i], u[i + 1], r_center, r_right, k, a_scale(t, alpha), a_scale_t(t,alpha),c)
            F_i_minus_1_2 = hll_flux(u[i - 1], u[i], r_left, r_center, k, a_scale(t, alpha), a_scale_t(t,alpha),c)

            source_term = u[i] * (1 - (u[i]**2 / c**2)) * a_scale_t(t, alpha) / a_scale(t, alpha)
            u_new[i] = u[i] - (dt / dr) * (F_i_plus_1_2 - F_i_minus_1_2) - dt * source_term
                
        u = u_new.copy()
        
        U.append(u)
    
    return U


#---------------------------------------------------------------------------------------------------------------
# PRIMARY (GODUNOV'S) SUPPORT FUNCTIONS
#---------------------------------------------------------------------------------------------------------------

# Function to calculate the flux
def calculate_flux(u, r, k, a_t):
    return (1 - k * r**2)**0.5 * u**2 / (2 * a_t)

# Function to calculate wave speeds
def calculate_wave_speeds(u, r, k, c, a, a_t):
    wave_speed_1 = abs((1 - k * r**2)**0.5 * u / a)
    wave_speed_2 = abs(u * (1 - (u**2 / c**2)) * a_t / a)
    return wave_speed_1, wave_speed_2

# Precompute the maximum wave speeds over the domain and time interval
def compute_dt(u_initial, r, dr, k, c, alpha, t_max):
    '''
    Calculates the maximum wave speed in order to ensure dt is set to meet CFL stability conditions
    '''
    
    max_speed = 0
    t_values = np.linspace(0.00001, t_max, 100)  # Discretize the time interval

    for t in t_values:
        a = a_scale(t,alpha)
        a_t = a_scale_t(t,alpha)
        for i in range(len(u_initial)):
            wave_speed_1, wave_speed_2 = calculate_wave_speeds(u_initial[i], r[i], k, c, a, a_t)
            max_speed = max(max_speed, wave_speed_1, wave_speed_2)

    dt = dr/max_speed
    return dt

# Define the HLL flux calculation with correct r interface handling
def hll_flux(u_left, u_right, r_left, r_right, k, a, a_t,c):
    r_interface = (r_left + r_right) / 2
    f_left = calculate_flux(u_left, r_interface, k, a_t)
    f_right = calculate_flux(u_right, r_interface, k, a_t)
    s_left, s_right = calculate_wave_speeds(u_left, r_interface, k, c, a, a_t)
                      
    if s_left >= 0:
        return f_left
    elif s_right <= 0:
        return f_right
    else:
        return (s_right * f_left - s_left * f_right + s_left * s_right * (u_right - u_left)) / (s_right - s_left)

#---------------------------------------------------------------------------------------------------------------
# SUPPORT FUNCTIONS
#---------------------------------------------------------------------------------------------------------------

# FLRW Acceleration Terms - First Order
def a_scale(t, alpha, a0=1, t0=1):
    return (a0 * alpha) + (t/t0)**alpha

def a_scale_t(t, alpha, a0=1, t0=1):
    return a0 + alpha * (t/t0)**(alpha - 1)

# Generate Numerical Meshes (Fixed)
def generate_mesh_d(dt,dx,t_max,L):
    '''
    Generates a mesh given dt and dx.
    
    Can do stability check outside of this function (CFL)
    '''
    Nx = int(L/dx)
    Nt = int(t_max/dt)

    x_grid = np.linspace(0,L,Nx+1) # two boundary nodes at 0 & L, rest are inner nodes
    t_grid = np.linspace(0,t_max,Nt+1) # Need to change node here by +1 for proper t-intervals
    return Nt, Nx, t_grid, x_grid

def generate_mesh_N(Nt, Nx, t_max, L):
    '''
    Generates a mesh given desired number of analysis nodes for x and t (Nx, Nt).
    
    Can do stability check outside of this function (CFL)
    '''
    dx = L/Nx
    dt = t_max/Nt

    x_grid = np.linspace(0,L,Nx+1) # two boundary nodes at 0 & L, rest are inner nodes
    t_grid = np.linspace(0,t_max,Nt+1) # Need to change node here by +1 for proper t-intervals
    
    return dt, dx, t_grid, x_grid

# Build u0 for explicit methods
def build_u0(L, dr, u_0_x, u_t_0, u_t_L):
    Nr = int(L/dr)
    r_grid = np.linspace(0,L,Nr+1)
    
    u0 = [u_0_x(r) for r in r_grid]
    
    u0[0] = u_t_0(0)
    u0[-1] = u_t_L(1)
    
    return r_grid, u0

# PDE U vs time Plotting Function
def PDE_plotter_1D(U, L, t_max, steps, t_min_plot, t_max_plot, x_min_plot, x_max_plot, style = 'multi', alpha_decay = .7):
    '''
    Function takes in calculated 1D data changing over time
    array of lists, the outer array holding timesteps and the inner holding the u value at each point for that step

    ! - Lots of Organizing Needed Here
    '''
    # Check for Errors
    if t_max_plot > t_max:
        raise Exception('Plotting end time exceeds maximum time')
    if t_min_plot < 0:
        raise Exception('Plotting start time must be greater than 0')
    if t_min_plot >= t_max_plot:
        raise Exception('Max plot time greater than minimum plot time')
    if x_min_plot < 0:
        raise Exception('Minimum x plot less than left boundary')
    if x_max_plot > L:
        raise Exception('Mamimum x plot larger than the right boundary')
    if x_min_plot >= x_max_plot:
        raise Exception('Invalid x value plotting range')
    

    dt = t_max/len(U)
    t_min_index = int(t_min_plot / dt)
    #*#t_max_index = int(t_max_plot / dt)
    t_max_index = int(t_max_plot / dt) +1
    print('t_min_index:',t_min_index)
    print('t_max_index:',t_max_index)
    
    dx = L/len(U[0])
    x_min_index = int(x_min_plot / dx)
    x_max_index = int(x_max_plot / dx)
    print('x_min_index:',x_min_index)
    print('x_max_index:',x_max_index)

    # Create slice to analyze
    U_slice = U[t_min_index:t_max_index]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('x')  # Set x-axis label
    ax.set_ylabel('T')  # Set y-axis label

    # Set the y-axis limits
    y_min = min([min(u) for u in U_slice]) - 0.5  # Find the minimum y-value in U with some buffer
    y_max = max([max(u) for u in U_slice]) + 0.5  # Find the maximum y-value in U with some buffer
    ax.set_ylim(y_min, y_max)

    # Set the x-axis limits
    ax.set_xlim(x_min_plot, x_max_plot)
    
    lines = []  # List to store line objects

    plt.show()
    
    interval = len(U_slice)//steps

    color_counter = 5
   
    for i in np.arange(0, len(U_slice), interval):
        # Update the alpha values of existing lines
        
        for line in lines:
            line.set_alpha(line.get_alpha() * alpha_decay)
    
        # Compute scaled x-axis values
        x_values = np.linspace(0, L, len(U_slice[i]))
        x_values_used = x_values[x_min_index:x_max_index+1]
        
        # Add new line
        if style == 'heat':
            line_color = 'red'
        if style == 'multi':
            line_color = ['cyan','green','yellow','magenta','blue'][color_counter%5]
            color_counter += 1

        new_line, = ax.plot(x_values_used,U_slice[i][x_min_index:x_max_index+1], color=line_color, alpha=1)  # Start with full opacity
        lines.append(new_line)  # Store the new line object
        #ax.set_title(f"Plot at time = {i/(len(U_slice)-1)}s")  # Update the title with the current step
        ax.set_title(f"Plot at time = {round(t_min_plot + i*dt,3)}s")
        
        
        # Handling plot display
        clear_output(wait=True)  # Clear the previous plot
        display(fig)  # Display the current figure

        time.sleep(0.25)  # Pause for quarter a second before the next update 




#===============================================================================================================
# ARCHIVE
#===============================================================================================================

def first_order_godunov_burgers_flrw(dt, dx, t_max, L, alpha, k, u_0_x, u_t_0, u_t_L):
    # Generate Mesh
    Nt, Nx, t_grid, x_grid = generate_mesh_d(dt,dx,t_max,L)    
    
    # Create u0 (needs u_0_x)
    u0 = u_0_x(x_grid)
    u0[0] = u_t_0(0)
    u0[-1] = u_t_L(0)

    # Create data holder, input initial conditions (u0)
    u = np.zeros((Nt+1, Nx+1)) 
    u[0, :] = u0

    # Time stepper
    for t in range(Nt):
        u_new = np.copy(u[t, :])

        # Iterative solver for the implicit scheme (using a simple fixed-point iteration or Newton's method)
        tol = 1e-6
        max_iter = 100
        for _ in range(max_iter):
            u_half = (u[t, :] + u_new) / 2
            t_half = (t_grid[t] + t_grid[t + 1]) / 2

            # Compute b, g, and S at time step n+1/2
            b_j_plus_1_2 = (1 - k*(u_half[1:])) / at(alpha,t_half)
            b_j_minus_1_2 = (1 - k*(u_half[:-1])) / at(alpha,t_half)

            # Apply minmod to compute limited slopes
            delta_u_plus = u_half[1:] - u_half[:-1]
            delta_u_minus = u_half[:-1] - u_half[1:]

            limited_slope_plus = np.array([minmod(delta_u_plus[i], delta_u_plus[i-1]) for i in range(1, len(delta_u_plus))])
            limited_slope_minus = np.array([minmod(delta_u_minus[i], delta_u_minus[i+1]) for i in range(len(delta_u_minus)-1)])

            g_j_plus_1_2 = f(u_half[:-1], u_half[1:] + np.concatenate(([0], limited_slope_plus)))
            g_j_minus_1_2 = f(u_half[1:], u_half[:-1] - np.concatenate((limited_slope_minus, [0])))

            S_j_half = S(u_half, t_half,at,at_t,alpha)

            # Update v_new
            u_new[1:-1] = u[t, 1:-1] - (dt / dx) * (b_j_plus_1_2[:-1] * g_j_plus_1_2[:-1] - b_j_minus_1_2[1:] * g_j_minus_1_2[1:]) + dt * S_j_half[1:-1]

            # Check for convergence
            if np.linalg.norm(u_new - u_half) < tol:
                break

        u[t + 1, :] = u_new
        
    return u


def second_order_godunov_burgers(u_0_x, u_t_0, u_t_L, a_t, dx, dt, t_max, L, k):
    """Solve the inviscid Burgers' equation using a second-order Godunov scheme with MUSCL and scale factor"""
    nx = int(L/dx) + 1

    x_grid = np.linspace(0,L,nx)
    
    u_0 = u_0_x(x_grid)
    
    nt = int(t_max / dt)
    nx = len(u_0)


    u = np.zeros((nt + 1, nx))
    u[0, :] = u_0

    for n in range(nt):
        t = n * dt
        a = a_t(t)
        u_n = u[n, :]
        uL = np.zeros(nx - 1)
        uR = np.zeros(nx - 1)
        flux = np.zeros(nx)

        for i in range(1, nx - 1):
            duL = minmod(u_n[i] - u_n[i - 1], u_n[i + 1] - u_n[i])
            duR = minmod(u_n[i + 1] - u_n[i], u_n[i + 2] - u_n[i + 1] if i + 2 < nx else 0)

            uL[i] = u_n[i] + 0.5 * duL
            uR[i] = u_n[i + 1] - 0.5 * duR

        for i in range(1, nx - 1):
            flux[i] = f(uL[i], uR[i], a)

        for i in range(1, nx - 1):
            u[n + 1, i] = u_n[i] - dt / dx * (flux[i] - flux[i - 1]) - dt * k * u_n[i] / a**2
        
        # Boundary conditions
        u[n + 1, 0] = u_t_0(0)
        u[n + 1, -1] = u_t_L(0)

    return u

# Godunov Flux
def f(ul, ur):
    flux = np.zeros_like(ul)
    
    # Conditions where ul > ur
    mask = ul > ur
    flux[mask & (ul > 0) & (ur > 0)] = 0.5 * ul[mask & (ul > 0) & (ur > 0)] ** 2
    flux[mask & (ul < 0) & (ur < 0)] = 0.5 * ur[mask & (ul < 0) & (ur < 0)] ** 2
    flux[mask & (ul > 0) & (ur < 0)] = 0

    # Conditions where ul <= ur
    mask = ul <= ur
    flux[mask & ((ul + ur) > 0)] = 0.5 * ul[mask & ((ul + ur) > 0)] ** 2
    flux[mask & ((ul + ur) <= 0)] = 0.5 * ur[mask & ((ul + ur) <= 0)] ** 2

    return flux
        
        
# Gudonov Support Function
def minmod(a, b):
    '''
    Minmod function for slope limiting
    '''
    if a * b <= 0:
        return 0
    else:
        return min(abs(a), abs(b)) * np.sign(a)


# Define the function S_j
def S(v, t, at, at_t, alpha):
    a_val = at(alpha,t)
    a_t_val = at_t(alpha,t)
    return - (v * (1 - v**2) * (a_t_val / a_val))