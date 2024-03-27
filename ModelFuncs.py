# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:45:18 2024

@author: ejfar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.constants as cons 
import pandas as pds
from PIL import Image
import io


def inchcon(length):
    """
    Function to convert inches to m 

    Parameters
    ----------
    length : array
        Array of dimensions in inches to be converted.

    Returns
    -------
    newlength : array
        Array of new dimensions in m

    """
    newlength = length*0.0254
    return(newlength)

def movemirrdoublet(lamda, f, D):
    """
    Function to find the lattice periodicity for the accordion setup which 
    involves translational movement of mirror M1 (see onenote diagrams). The 
    lattice spacing is proportional to the focal length of the lens used and 
    inversely proportional to the distance between the beams incoming to the 
    lens.

    Parameters
    ----------
    lamda : float
        Wavelength of beams.
    f : float
        Focal length of lens.
    D : float
        Distance between incoming parallel beams.

    Returns
    -------
    a : float
        The spacing fo fringes in the lattice.

    """
    a = lamda * f / D
    return(a)

def movemirrasph(lamda, f, D):
    
    a = lamda /(D*np.sqrt((D/2)**2 + f**2))
    return(a)

def omegaLam(lamda):
    """
    Function to calculate angular frequency from wavelength

    Parameters
    ----------
    lamda : float
        wavelength.

    Returns
    -------
    Omega : float
            Angular frequency.

    """
    Omega = 2*np.pi*cons.c/lamda
    return(Omega)

def intensity(f, theta, lamda, W0, step):
    """
    Function to calculate the intensity from the interference of two Gaussian 
    beams through their electric fields when passing from collimated beams 
    through a focusing lens

    Parameters
    ----------
    f : float
        focal length of lens.
    theta : float
        angle between beams in radians.
    lamda : float
        Wavelength of beams.
    W0 : float
        Radius of collimated beams before the lens.
    step : float
        Number of array points caluclation is done over, effectively a resolution.


    Returns
    -------
    y : array
        The y co-ordinates of the region of interest.
    x : array
        The x co-ordinates of the region of interest.
    I : array
        The 2D array of intensity values across the area y x x

    """
    
    # define wavenumber
    k = 2*np.pi/lamda
    
    # create k vector
    kvec_1 = k * np.array([0, 0, 1])
    kvec_2 = kvec_1
    
    omega = omegaLam(lamda)
    # define time at which E fields are evaluated
    t = 0

    # define ROI through how far into each axis the image "zooms"
    zzoom = 0.005
    xzoom = 0.05
    yzoom = xzoom
    
    # create co-ordinate arrays and define propagation step
    z = np.linspace(-f*zzoom, f*zzoom, step)
    dz = z[1] - z[0]
    x = np.linspace(-W0*xzoom, W0*xzoom, step)
    y = np.linspace(-W0*yzoom, W0*yzoom, step)
    
    # create hold arrays
    E1 = np.zeros((step, step), dtype=complex)
    E2 = np.zeros_like(E1)
    
    # define the (1/q) parameters before and after the lens
    q_in_1 = 1j * np.pi * W0**2 / lamda 
    q_in_2 = 1j * np.pi * W0**2 / lamda 
    
    q_out_1 = q_in_1 / (-1/f * q_in_1 + 1) + (f-zzoom*f) # additional term makes
                                                        # jump along z to ROI
    q_out_2 = q_in_2 / (-1/f * q_in_2 + 1) + (f-zzoom*f)
    
    # iterate through each co-ordinate
    for p in range(len(z)):
        # In free space, new q parameter at each z point is q after lens + z
        # Each beam has own frame of reference
        dz_1 = dz * np.cos(theta)
        dz_2 = dz_1
        
        q_out_1 += dz_1
        q_out_2 += dz_2
        
        inq_1 = 1/q_out_1
        inq_2 = 1/q_out_2
        
        # define z position of "camera" (plane of view)
        if p == int(len(z)/2):
        
            for m in range(len(x)):
                for yy in range(len(y)):
                
                    # co-ordinate transformation
                    xp_1 = z[p] * np.sin(theta) + x[m] * np.cos(theta)
                    zp_1 = z[p] * np.cos(theta) - x[m] * np.sin(theta)
                    xp_2 = -z[p] * np.sin(theta) + x[m] * np.cos(theta)
                    zp_2 = z[p] * np.cos(theta) + x[m] * np.sin(theta)
                    
                    # define cylindrical co-ordinate
                    rho_sq_1 = xp_1**2 + y[yy]**2
                    rho_sq_2 = xp_2**2 + y[yy]**2
                    
                    r_1 = np.array([xp_1, y[yy], zp_1])
                    r_2 = np.array([xp_2, y[yy], zp_2])
            
                    E1[m, yy] = inq_1 * np.exp(-1j*k*rho_sq_1/2 * inq_1) * np.exp(1j * np.dot(kvec_1, r_1) - omega*t)
                    E2[m, yy] = inq_2 * np.exp(-1j*k*rho_sq_2/2 * inq_2) * np.exp(1j * np.dot(kvec_2, r_2) - omega*t) * np.exp(1j*np.pi)
    
    # Intensity = |E|^2
    I = np.absolute(E1 + E2)**2
    
    return(y, x, I)



def surf2subplot(X, Y, Z1, Z2, step, text):
    """
    Function to automatically generate 2D subplot images

    Parameters
    ----------
    X : array
        The x axis values.
    Y : array
        The y axis values.
    Z1 : 2D array
        The z axis values (the 2D magnitudes) of the first subplot.
    Z2 : 2D array
        The z axis values (the 2D magnitudes) of the second subplot.
    step : float
        Number of array points caluclation is done over, effectively a resolution.
    text : array of strings
        In following order: Figure title, ylabel, xlabel, 1st subplot title, 
        2nd subplot title, colorbar label.

    Returns
    -------
    None.

    """
    # plt.contourf transposes image
    Z1 = np.transpose(Z1)
    Z2 = np.transpose(Z2)
    
    fig, axs = plt.subplots(1, 2, layout='constrained')
    fig.suptitle(text[0], fontsize=48)
    plt.tick_params(axis='both', labelsize=30)
    
    c_1 = axs[0].contourf(Y, X, Z1, step, cmap=cm.magma, labelsize=30)
    axs[0].set_xlabel(text[2], fontsize = 40)
    axs[0].set_ylabel(text[1], fontsize=40)
    axs[0].set_title(text[3], fontsize=44)
    # cbar1 = plt.colorbar(c_1, ax=axs[0])
    axs[0].tick_params(labelsize=30)
    
   
    c_2 = axs[1].contourf(Y, X, Z2, step, cmap=cm.magma, labelsize=30)
    # c_2.tick_params(labelsize=30)
    axs[1].set_xlabel(text[2], fontsize=40)
    axs[1].set_ylabel(text[1], fontsize=40)
    axs[1].set_title(text[4], fontsize=44)
    # use colorbar for righthand plot
    cbar = fig.colorbar(c_2, ax=axs[1])
    cbar.set_label(text[5], fontsize=40)
    cbar.ax.tick_params(labelsize=30)
    
    plt.show()
    
    return()
