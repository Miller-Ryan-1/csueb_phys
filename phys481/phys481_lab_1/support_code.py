import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt

# Solving the Newton's law model and also equations 17a and 17b from Lorenceau paper:

# Newton's law model
def DZ_dt_Newton(Z, t, args):
    h = args[0]
    g = args[1]
    b = args[2]
    return [Z[1], -Z[1]**2/Z[0] - g +g*h/Z[0] -b*Z[1]/Z[0] ]

# Lorenceau model:
def DZ_dt_Lor(Z, t, args):
    h = args[0]
    g = args[1]
    Omeg = args[3]
    if Z[1]>0:
        return [Z[1], 1/Z[0] - 1 -Omeg*Z[1] - (Z[1])**2/Z[0]]
    else:
        return [Z[1], 1/Z[0] - 1 -Omeg*Z[1] ]

def plot_osc(df,h=10.0, g=9.8e2,b = 23,factor=2.60):

    # prepare data for plotting:
    z_data1 = df['z']+h # change the overall level so that bottom of straw is z=0
    time_axis1 = df['t']  # only include data for positive times (after cap is released)
    
    # prepare parameters for solving models:
    Omeg = 0.062*factor
    params = (h,g,b,Omeg)    
    
    # solve Newton model:
    t_soln = time_axis1
    Z_soln_Newton = sp.integrate.odeint(DZ_dt_Newton, [0.02, 0], t_soln, args=(params,))   

    z_soln_Newton = Z_soln_Newton[:,0]      # fluid height

    # solve Lorenceau model, equation 17a and 17b....
    t_solnLor = np.arange(0, 30, 0.01)
    Z_soln_Lor = sp.integrate.odeint(DZ_dt_Lor, [0.02, 0.00], t_solnLor, args=(params,))   
    
    z_soln_Lor = Z_soln_Lor[:,0]*h       # fluid height

    Omeg = 0.062*factor
    params = (h,g,b,Omeg)

    plt.clf()
    plt.plot(time_axis1,z_data1,'b.',label='Data') 
    plt.plot(t_soln,z_soln_Newton,'r',label='Newtonian model')

    plt.xlabel('time (sec)',fontsize=15)
    plt.ylabel('fluid level (cm)',fontsize=15)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    plt.plot(t_solnLor*(h*1e-2/9.8)**0.5,z_soln_Lor,'g--',label='Lorenceau model')
    plt.legend(frameon=False,loc=1)
    plt.xlim([-0.2,3])
    plt.grid()
    plt.show()

Y_FFT = np.fft.rfft(df['z'])
fft_power = pd.DataFrame(Y_FFT * np.conjugate(Y_FFT))

DT = df.index[1]-df.index[0]   # sample time
fft_power.index=(np.fft.fftfreq(df['z'].shape[0])/DT)[0:len(fft_power)]
fft_power.plot()
plt.xlim([0,.5])

title = 'Power Spectrum of Straw Oscillations'
xaxis_label = 'frequency (Hz)'
yaxis_label = 'signal (a.u.)'
plt.axvline(x=.067)
plt.xlabel(xaxis_label,fontsize=15)
plt.ylabel(yaxis_label,fontsize=15)
plt.grid()
plt.title(title,fontsize=20)