# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:41:09 2024

@author: ejfar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize as opt
from PIL import Image
import fittingfuncs as ff

plt.close('all')

# define the length corresoponding to 1 pixel
pixel_size = 5.2  #um

# load image

image0 = Image.open("Intensity, theta0.5.tiff")
# take mean across layers and normalise intensity
image = np.mean(image0, axis=2)
image = image / image.max()
# create a matching title string to use in the final plot
title_str = 'Fit to Fringes Sample Data: f=100mm, W0=0.5mm'


#%% - Scale and Recenter

# new x and y arrays which are scaled by the size of 1 pixel and centered on 0
yr = np.linspace(-image.shape[0]/2, image.shape[0]/2, image.shape[0]) * pixel_size
xr = np.linspace(-image.shape[1]/2, image.shape[1]/2, image.shape[1]) * pixel_size

X, Y = np.meshgrid(xr, yr)

# Check the plot
plt.figure()
c1 = plt.contourf(X, Y, image, 100, cmap=cm.magma)
plt.xlabel("x (um)")
plt.ylabel("z (um)")
plt.title("Original Image Re-centered and scaled to Pixel Length")
cbar1 = plt.colorbar(c1)
cbar1.set_label("Intensity")



#%% - Slicing

# Take Middle 2D Slices about point of highest intensity to estimate initial  
# guess parameters

# find the max intensity point to slice about
slice_val = image.max()
slice_point = np.where(image==slice_val)[1]
slice_row, slice_column = np.where(image==slice_val)

# take x slice through row of maximum intensity
x_slice = image[slice_row[0], :]
x_fit = np.arange(0, len(x_slice))

# Estimate fit parameters and fit along x

plt.figure()
plt.plot(xr, x_slice)
plt.xlabel("x (um)")
plt.ylabel("Intensity ()")
plt.title("Slice through central lattice maximum")

# fit parameters (A, k, phi, x0, s, b)
k_g = 2*np.pi/246 # read from graph as no. pixels between peaks
phi_g = 0
sx_g = 800
b_g = 75
A_g = float(x_slice.max()) - b_g
x0_g = -54 #  np.where(x_slice==x_slice.max())[0][0]


# fitting model takes parameters in following order
p0x = [A_g, k_g, phi_g, x0_g, sx_g, b_g]

# pass to scipy fit function
A_f, k_f, phi_f, x0_f, sx_f, b_f = opt.curve_fit(ff.sinefit, xr, x_slice, p0x)[0]


# Plot fit data
xfitdata = ff.sinefit(xr, A_f, k_f, phi_f, x0_f, sx_f, b_f)
plt.plot(xr, xfitdata, 'r-')
plt.legend(["Data Slice", "Fit"], loc='best')

# Slice and repeat along y
y_slice = image[:, slice_column[0]]
y_fit = np.arange(0, len(y_slice))
y_slice = image[:, slice_column[0]]

plt.figure()
plt.plot(yr, y_slice)
plt.xlabel("y (pixel)")
plt.ylabel("Intensity ()")
plt.title("Slice through central fringe")

y0_g = 0 
sy_g = 20
b_y = 80
p0y = [A_f, y0_g, sy_g, b_y]
Afy, y0_f, sy_f, by_f = opt.curve_fit(ff.expfit, yr, y_slice, p0y)[0]

yfitdata = ff.expfit(yr, Afy, y0_f, sy_f, by_f)
plt.plot(yr, yfitdata, 'r-')
plt.legend(["Data Slice", "Fit"], loc='best')

# Create array to hold all guess parameters
# [A,   k,   phi, x0,  y0, sigmax, sigmay, off]
passparams = np.array([A_f, k_f, phi_f, x0_f, y0_f, sx_f, sy_f, b_f])

# call to the 2D fitting function, which plots the fit in-function
# return the overall result object and the fit report 
img, fitrep = ff.fitimagesin(image, pixel_size, title_str, passparams)

