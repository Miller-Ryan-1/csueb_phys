import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from IPython.display import display, clear_output


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
            flux[i] = godunov_flux(uL[i], uR[i], a)

        for i in range(1, nx - 1):
            u[n + 1, i] = u_n[i] - dt / dx * (flux[i] - flux[i - 1]) - dt * k * u_n[i] / a**2
        
        # Boundary conditions
        u[n + 1, 0] = u_t_0(0)
        u[n + 1, -1] = u_t_L(0)

    return u


# Determing the FLRW acceleration term
def a(t):
    '''
    Using the standard a(t) = t/t0^(2/3) for matter dominated universe.

    Parameters: t = time since beginning.
    '''
 
    return 1 + t**(2/3)

def godunov_flux(ul, ur, a):
    '''
    Godunov's flux functions for the inviscid Burgers'
    
    ! - Cite Source!
    '''
    if ul > ur:
        if ul > 0 and ur > 0:
            return 0.5 * ul**2 / a
        elif ul < 0 and ur < 0:
            return 0.5 * ur**2 / a
        else:
            return 0
    else:
        if ul + ur > 0:
            return 0.5 * ul**2 / a
        else:
            return 0.5 * ur**2 / a
        
def minmod(a, b):
    '''
    Minmod function for slope limiting - improves Gudanov's w/ MUSCL - Monotonic Upstream Centered Schemes for Conservation Laws
    '''
    if a * b <= 0:
        return 0
    else:
        return min(abs(a), abs(b)) * np.sign(a)
    
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

        time.sleep(0.25)  # Pause for half a second before the next update 