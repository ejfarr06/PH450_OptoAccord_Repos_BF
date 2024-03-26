# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:01:54 2024

@author: ejfar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import lmfit
from lmfit import minimize, Parameters,Model
from lmfit.models import LorentzianModel,GaussianModel,ConstantModel


def sinefit(x, A, k, phi, x0, s, b):
    y = A * np.sin(k*x + phi)**2 * np.exp(-(x-x0)**2 / (2*s**2)) + b
    return(y)

def sinlorenv(x, A, k, phi, x0, s, b):
    p = (x-x0) / (s/2)
    y = A * np.sin(k*x + phi)**2 * (1 / (1+p**2)) + b
    return(y)

def expfit(x, A, data0, sigma, b):
    output = A * np.exp(-(x - data0)**2 / (2*sigma**2)) + b
    return(output)

def lorentz(x, A, x0, sigmax, b):
    p = (x-x0) / (sigmax/2)
    L = A / (1+p**2) + b
    return(L)

def sinlorzfit(x_vals, A, k, phi, x0, y0, sigmax, sigmay, off):
    x, y = x_vals
    p = (y-y0) / (sigmay/2)
    z = A*np.sin(k * x + phi)**2 *np.exp(-(x-x0)**2 / (2*sigmax**2)) * (1 / ((1+p**2)))  + off
    
  
    # returning the output.ravel important for the array handling in the lmfit 
    # functions
    return(z.ravel())

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

