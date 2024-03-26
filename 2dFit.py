# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:41:09 2024

@author: ejfar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from lmfit import minimize, Parameters,Model
import scipy.optimize as opt
from PIL import Image



def sinefit(x, A, k, phi, x0, s, b):
    y = A * np.sin(k*x + phi)**2 * np.exp(-(x-x0)**2 / (2*s**2)) + b
    return(y)

def expfit(x, A, data0, sigma, b):
    output = A * np.exp(-(x - data0)**2 / (2*sigma**2)) + b
    return(output)

def singaussfit(x_vals, A, k, phi, x0, y0, sigmax, sigmay, off):
    x, y = x_vals
    z = A*np.sin(k * x + phi)**2 *np.exp(-(x-x0)**2 / (2*sigmax**2))* np.exp(-(y-y0)**2 / (2*sigmay**2))  + off


    # returning the output.ravel important for the array handling in the lmfit 
    # functions
    return(z.ravel())

def fitimagesin(ImageOG, camera, title_str, pass_parameters):
    
    # Step 1: Convert to um***************************
    # define new x and y with 0 at plot centre and scaled by pixel length in um
    # use cropped image size
    y = np.linspace(-ImageOG.shape[0]/2, ImageOG.shape[0]/2, ImageOG.shape[0]) * camera
    x = np.linspace(-ImageOG.shape[1]/2, ImageOG.shape[1]/2, ImageOG.shape[1]) * camera
    
    
    x, y = np.meshgrid(x, y)
    
    # Use lmfit's Model function to define the fitting model as the sine function
    # label it for future parameter output readouts
    Sine_mod1 = Model(singaussfit, prefix='Sin_Mod_')
    # Define initial pass parameters
    # NOTE: All variable parameters in fit function must be defined here. 
    # If not, fit module assigns undefined initial parameters to infinity, which
    # create NaN values and terminates the fit process
    
    # set up the initial guess parameters and convert to um
    #             [A,   k,   phi, x0,  y0, sigmax, sigmay, off]
    # ip = pass_parameters * camera
    iniguess = Sine_mod1.make_params(A=pass_parameters[0], k = pass_parameters[1], 
                                     phi=pass_parameters[2], x0=pass_parameters[3], 
                                     y0=pass_parameters[4], sigmax=pass_parameters[5], 
                                     sigmay=pass_parameters[6], off=pass_parameters[7])
   
    
    
    # Fit to the data
    Sin_fit1 = Sine_mod1.fit(ImageOG.ravel(), iniguess, x_vals = (x, y))
    
    print(Sin_fit1.fit_report())
    resu = Sin_fit1.fit_report() #.valuesdict()
    # create new arrays to model the fit data
    a = np.linspace(-ImageOG.shape[0]/2*camera, ImageOG.shape[0]/2*camera, 100)
    b = a
    a, b = np.meshgrid(a, b)
    # get out optimized variable parameters
    fitted = Sin_fit1.eval(x_vals=(a, b))
    
    # Plot both original (cropped) image and the fitted curve on top
    fig, axs = plt.subplots(1, 1)
    fig.suptitle(title_str)
    axs.set_xlabel("x (um)")
    axs.set_ylabel("y (um)")
    axs.legend(["Image", "Fit Data"], loc='best')
    c = axs.imshow(ImageOG, cmap=cm.magma, extent=(x.min(), x.max(), y.min(), y.max()))
    cbar = fig.colorbar(c)
    cbar.set_label("Intensity")
    
    axs.contour(a, b, fitted.reshape(a.shape), 10, colors='w')
    
    plt.show()
    
    
    return(Sin_fit1, resu)

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
A_f, k_f, phi_f, x0_f, sx_f, b_f = opt.curve_fit(sinefit, xr, x_slice, p0x)[0]


# Plot fit data
xfitdata = sinefit(xr, A_f, k_f, phi_f, x0_f, sx_f, b_f)
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
Afy, y0_f, sy_f, by_f = opt.curve_fit(expfit, yr, y_slice, p0y)[0]

yfitdata = expfit(yr, Afy, y0_f, sy_f, by_f)
plt.plot(yr, yfitdata, 'r-')
plt.legend(["Data Slice", "Fit"], loc='best')

# Create array to hold all guess parameters
# [A,   k,   phi, x0,  y0, sigmax, sigmay, off]
passparams = np.array([A_f, k_f, phi_f, x0_f, y0_f, sx_f, sy_f, b_f])

# call to the 2D fitting function, which plots the fit in-function
# return the overall result object and the fit report 
img, fitrep = fitimagesin(image, pixel_size, title_str, passparams)

