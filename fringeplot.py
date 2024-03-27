# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:58:25 2024

@author: ejfar
"""

import numpy as np
import scipy.constants as cons 
import pandas as pds
import ModelFuncs as mf


# define approximate asphere lens parameters
f = np.array([100, 250, 500])*1e-3
dia = np.ones((1, 3)) * mf.inchcon(1)
NA = np.ones((1, 3)) * 0.5

# define beam properties
lam = 532e-9
omega = mf.omegaLam(lam)
W0 = np.array([1e-3])

# define resonances for Rb and K for temperature conversion
lam41KD = np.array([770.1079e-9, 766.7005e-9])
omega_41KD = mf.omegaLam(lam41KD)

omega_K = np.mean(omega_41KD)

lam87RbD = np.array([780.2412, 794.9788])
omega_87RbD = mf.omegaLam(lam87RbD)

omega_Rb = np.mean(omega_87RbD)

# define detuning
detun_K = omega - omega_K 
detun_Rb = omega - omega_Rb

# define scattering rate
scatrt = (cons.e**2 * omega**2) / (6*np.pi*cons.epsilon_0*cons.electron_mass*cons.c**3)

# define step size (effectively resolution)
step = 101

# set system limits min and max beam separaration
d_min = 2*W0*1.1
d_max = np.arcsin(NA)

# separation can't be larger than the lens size of else no effect
d_max = np.clip(d_max, 0, mf.inchcon(1))

# for every lens l
for l in range(len(f)):
    
    fp = f[l]   
    theta_max = np.arctan(d_max[0, l] / fp)
    
    # for every beam width wo
    for wo in range(len(W0)):
        theta_min = np.arctan(d_min[wo] / fp)

        ymx, xmx, I_thmax = mf.intensity(fp, theta_max, lam, W0[wo], step)
        ymn, xmn, I_thmin = mf.intensity(fp, theta_min, lam, W0[wo], step)  
        
        I_thmax = I_thmax / I_thmax.max()
        I_thmin = I_thmin / I_thmin.max()


        # Convert to trap depth
        U_const = 3*np.pi*cons.c**2 / 2
        
        U_K_thmx = U_const / omega_K**3 * (scatrt / detun_K) * I_thmax
        U_K_thmn = U_const / omega_K**3 * (scatrt / detun_K) * I_thmin
        
        U_Rb_thmx = U_const / omega_Rb**3 * (scatrt / detun_Rb) * I_thmax
        U_Rb_thmn = U_const / omega_Rb**3 * (scatrt / detun_Rb) * I_thmin
        
        T_K_thmx = U_K_thmx / cons.Boltzmann
        T_K_thmn = U_K_thmn / cons.Boltzmann
        
        T_Rb_thmx = U_Rb_thmx / cons.Boltzmann
        T_Rb_thmn = U_Rb_thmn / cons.Boltzmann
        
        # Plot
        # define labels
        sup_I = "Normalised Intenisty for W0 = {:} mm, f = {:}mm".format(W0[wo]*1e3, fp*1e3)
        
        sup_K = "Potential for 41K, W0 = {:} mm, f = {:}mm".format(W0[wo]*1e3, fp*1e3)
        sup_Rb = "Potential for 87Rb, W0 = {:} mm, f = {:}mm".format(W0[wo]*1e3, fp*1e3)
        
        xlab = "x [um]"
        zlab = "y [um]"
        subt_1 = "Max Theta"
        subt_2 = "Min Theta"
        
        con_lab = 'Intensity'
        con_lab_temp = 'Trap Depth (K)'
        
        finam_I = "Intensity, theta{:.1f} W0{:.1f} f{:.1f}".format(theta_max, W0[wo]*1e3, fp*1e3)
        
        text_I = np.array([sup_I, zlab, xlab, subt_1, subt_2, con_lab])
        text_K = np.array([sup_K, zlab, xlab, subt_1, subt_2, con_lab_temp])
        text_Rb = np.array([sup_Rb, zlab, xlab, subt_1, subt_2, con_lab_temp])
        
        Iplot = mf.surf2subplot(ymx*1e6, xmx*1e6, I_thmax, I_thmin, step, text_I)
        # Kplot = surf2subplot(zmx*1e6, xmx*1e6, U_K_thmx, U_K_thmn, step, text_K)
        # Rbplot = surf2subplot(zmx*1e3, xmx*1e3, U_Rb_thmx, U_Rb_thmn, step, text_Rb)
        
        
    
        