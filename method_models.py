# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:42:30 2024

@author: ejfar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pds
import ModelFuncs as ms

#******************************************************************************
# Define constant variables
lam = 532e-9
step = 101

# Choose arbitrary waist for now, eventually will be array of values to find 
# optimum waist for trap (waist occurs at trap site)
# Waist is needed here to define minimum beam separation
W0 = 1e-3 # m


#%% Linear Mirror

f = np.linspace(50, 500, 10)*10**-3
Dmax = ms.inchcon(1) * 0.9
Dmin = W0*2*1.5
D = np.linspace(Dmin, Dmax, 100)

plt.figure(1)
plt.xlabel("Beam Spacing, D (mm)", fontsize=40)
plt.ylabel("Lattice spacing, a (um)", fontsize=40)
plt.title("Linear Mech. Mirror Lattice Spacing", fontsize=48)
plt.tick_params(axis='both', labelsize=30)

for i in range(len(f)):
    a = ms.movemirrdoublet(lam, f[i], D)
    plt.plot(D*10**3, a*10**6, label='f={:.0f}mm'.format(f[i]*10**3))
plt.legend(loc='best', fontsize=28)


#%% Telescope

# Load ETL data from optotune
dims = pds.read_excel('ETLData.xlsx')
dims_np = dims.to_numpy()

# Define lens number, each len's maximum focal length and clear aperture
lens = dims_np[:, 0]
f_in_tele = dims_np[:, 1]*10**-3
ca_in = dims_np[:, 2]*10**-3

# Begin plot
fig, axs = plt.subplots(1, 2, layout='constrained')
fig.suptitle("Telescope Method Parameters for Commercial ETLs", fontsize=44)
plt.tick_params(labelsize=30)
axs[0].set_xlabel("f1 (mm)", fontsize=40)
axs[0].set_ylabel("D (mm)", fontsize=40)
axs[0].set_title("Achieveable D ", fontsize=40)
axs[0].tick_params(labelsize=30)
axs[1].set_xlabel("Distance D (mm)", fontsize=40)
axs[1].set_ylabel("Lattice spacing (um)", fontsize=40)
axs[1].set_title("Lattice w/ 100mm Final Lens", fontsize=40)


# Hold array for the distances achieved by each lens pair
D_tele = np.zeros((step, len(lens)))
# Lattice created by third lens
# Choose f = 100mm as an approx. figure
f100 = 100e-3


for l in range(len(lens)):
    # f1+f2 always constant, so f2 is the reverse of f1
    f_1 = np.linspace(f_in_tele[l]*0.01, f_in_tele[l], step)
    f_2 = f_1[::-1]
    
    # From geometry
    D_i = ca_in[l]*f_2 / f_1
    # D cannot be bigger than maximum CA otherwise it will miss the lens and be
    # pointless
    for m in range(len(D_i)):
        if D_i[m] > ca_in[l]:
            D_i[m] = 0
        # D also cannot be smaller than 150% of the size of two beams
        if D_i[m] < 2*W0*1.5:
            D_i[m] = 2*W0*1.5
    
    axs[0].plot(f_1*10**3, D_i*10**3, label='Δf = {:.0f}mm'.format(f_in_tele[l]*10**3))
    axs[0].legend(loc='best', fontsize=30)
    
    # calculate lattice spacing using only the region where the light begins to
    # hit the lens
    a_tele = ms.movemirrdoublet(lam, f100, D_i[50:])
    axs[1].plot(D_i[50:]*10**3, a_tele*10**6, label = 'Lens{:}'.format(l+1))
    axs[1].legend(loc='best', fontsize=30)
    
    # record distances
    D_tele[:, l] = D_i

plt.show()
