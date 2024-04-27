# For solving
import numpy as np
import pandas as pd
import math

# For plotting
import matplotlib.pyplot as plt
import time
from IPython.display import display, clear_output

def heat_1d_explicit(f_u_0_x, D, dx, L, f_u_t_0, f_u_t_L, t_max, dt_mod=1):
    '''
    This function takes in an initial temperature distribution, along with boundary conditions,
    and emits the changes to the temperature distribution based on finite difference analysis.
    
    NOTE 1: This function is NOT optimized for numpy use in Python.  It works well for simple problems, but
    needs to be updated to use Numpy arrays for large datasets.
    
    Parameters:    
    f = the function governing the initial heat condition in the rod, must be a def f() [function]
    D = Heat diffusion constant (aka gamma) [float]
    dx = distance between analysis nodes [float]
    L = length of rod [float]
    u_t_0 = Boundary condition at zero [float]
    u_t_L = Boundary Condition at L [float]
    t_max = max time [integer or float]
    dt_mod = (optional) alter this to change then impact of the time interval in caase solutions are unstable
    
    Returns:
    U = array containing the temperature distributions across length L at time step t=0 to t=t_max
    '''
    
    # First set the time step such that dt </= dx**2/(2*D); this determines mu:
    dt = dx**2 / (2*D) # dt = .001, should be equal to or less than dx^2/(2*D) for stability requirements
    mu = D * (dt/dx**2) * dt_mod # Related to dt above, must be </= to .5
    
    # Generate x nodes across rod:
    Nx = int(L/dx)
    
    # Initialize Array:
    U = []

    # Create holder for first row of array = the initial temperature distribution:
    u_t_x_0 = []

    # Populate holder with initial distribution, need N+1 to include x = 3
    for i in range(Nx+1):
        # Normalize the step
        n = i*dx
        
        # Solve the initial value equation for each node:
        # This try/except statement is put in in case the initial condition is a constant:
        try:
            f_n = f_u_0_x(n)
        except:
            f_n = f_u_0_x
            
        u_t_x_0.append(f_n)
    
    # Replace initial distribution boundary conditions with, well, boundary conditions
    try:
        u_t_x_0[0] = f_u_t_0(0)
    except:
        u_t_x_0[0] = f_u_t_0
            
    try:
        u_t_x_0[-1] = f_u_t_L(0)
    except:
        u_t_x_0[-1] = f_u_t_L
    
    # Add this first row to the Array to return
    U.append(u_t_x_0)

    # Create range of time to examine
    t_steps = math.ceil(t_max/dt)

    # Run this loop the number of time steps you want to analyze
    for t in range(t_steps):

        # First, create empty time step - basically initize an empty list to fill with u(t,x) :
        u_t_x = []
        
        # Now grab the previous time step distribution (the value of u at each x for the previous t):
        u_line = U[-1]

        # Now loop through all 'inner' values, that is 0 < x < L, requiring a range of 1 to N:
        for i in range(1,Nx):
            # Using a modified equation 5.14 to calculate u at each x value:
            u = u_line[i] + mu*(u_line[i+1] + u_line[i-1] - 2*u_line[i])
            # Append this value to the time step line:
            u_t_x.append(u)

        # Append the first and last values with the boundary conditions.  Use try and except in case constant
        try:
            a = f_u_t_0(t)
        except:
            a = f_u_t_0
            
        try:
            b = f_u_t_L(t)
        except:
            b = f_u_t_L
        
        u_t_x.insert(0,a)
        u_t_x.insert(len(u_line),b)

        # Now append the full solutions matrix with time step distribution:
        U.append(u_t_x) 

    return U


def heat_1d_implicit(f_u_0_x, D, dx, dt, L, f_u_t_0, f_u_t_L, t_max):
    # Number of dx nodes, including 0 node
    Nx = int(L/dx) + 1

    # Discretize x and t
    x = np.arange(0,L+dx,dx)
    t = np.arange(0,t_max+dt,dt)

    # Create return array (n x m matrix)
    n = len(x)
    m = len(t)
    T = np.zeros((n,m))

    # Set initial conditions
    T[0,:] = f_u_0_x(x)

    # Set boundary values
    T[:,0] = f_u_t_0(t)
    T[:,-1] = f_u_t_L(t)

    # Derive the Lambda
    lambd = D * dt/dx**2

    # LHS = A: a tridiagnonal matrix
    A = np.diag([1+2*lambd]*(Nx-2)) + np.diag([-lambd]*(Nx-3),1) + np.diag([-lambd]*(Nx-3),-1)

    # Solve the equations
    for i in range(0,n-1):
        b = T[i,1:-1].copy() #Need the copy here!
        b[0] = b[0] + lambd*T[i+1,0]
        b[-1] = b[-1] + lambd*T[i+1,-1]
        sol = np.linalg.solve(A,b)
        T[i+1,1:-1] = sol

    return T


def crank_nicholson_1d(f_u_0_x, D, dx, dt, L, f_u_t_0, f_u_t_L, t_max):
    # Number of dx nodes, including 0 node
    Nx = int(L/dx) + 1

    # Discretize x and t
    x = np.arange(0,L+dx,dx)
    t = np.arange(0,t_max+dt,dt)

    # Create return array (n x m matrix)
    n = len(x)
    m = len(t)
    T = np.zeros((n,m))

    # Set initial conditions
    T[0,:] = f_u_0_x(x)

    # Set boundary values
    T[:,0] = f_u_t_0(t)
    T[:,-1] = f_u_t_L(t)

    # Derive the Lambda
    lambd = D * dt/dx**2

    # LHS = A: a tridiagnonal matrix
    A = np.diag([2+2*lambd]*(Nx-2)) + np.diag([-lambd]*(Nx-3),1) + np.diag([-lambd]*(Nx-3),-1)

    # RHS tridagonal matrix
    B = np.diag([2-2*lambd]*(Nx-2)) + np.diag([lambd]*(Nx-3),1) + np.diag([lambd]*(Nx-3),-1)

    # Solve the equations
    for i in range(0,n-1):
        b = T[i,1:-1].copy() #Need the copy here!
        b[0] = b[0] + lambd*T[i+1,0]
        b[-1] = b[-1] + lambd*T[i+1,-1]
        sol = np.linalg.solve(A,b)
        T[i+1,1:-1] = sol

    return T


def heat_plotter(U, t_max, steps):

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('x')  # Set x-axis label
    ax.set_ylabel('T')  # Set y-axis label

    # Set the y-axis limits
    y_min = min([min(u) for u in U]) - 0.5  # Find the minimum y-value in U with some buffer
    y_max = max([max(u) for u in U]) + 0.5  # Find the maximum y-value in U with some buffer
    ax.set_ylim(y_min, y_max)

    lines = []  # List to store line objects
    alpha_decay = .5  # Factor to reduce the alpha of previous lines

    plt.show()

    step_length = len(U)//steps

    for i in np.arange(0, len(U), step_length):
        # Update the alpha values of existing lines
        for line in lines:
            line.set_alpha(line.get_alpha() * alpha_decay)

        # Add new line
        new_line, = ax.plot(U[i], color='red', alpha=1.0)  # Start with full opacity
        lines.append(new_line)  # Store the new line object
        ax.set_title(f"Plot at time = {round((t_max*i)/(len(U)-1),4)}s")  # Update the title with the current step

        # Handling plot display
        clear_output(wait=True)  # Clear the previous plot
        display(fig)  # Display the current figure

        time.sleep(0.25)  # Pause for half a second before the next update