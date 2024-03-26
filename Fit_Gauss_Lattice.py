# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:30:38 2024

@author: ejfar
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from PIL import Image
import time
import latticefuncs as lf



plt.close('all')
start = time.monotonic()

# define the length corresoponding to 1 pixel
pixel_size = 1.55  #um

starttime = time.monotonic()
hrs = 0.1 # length of runtime in hours
stepmins = 1 # how many miuntes between images
runtime = 3600 * hrs #runtime in seconds
timestep = 60 * stepmins #timestep in seconds
endcounter = 1 #runtime / timestep
counter = 0

xdata = np.zeros((endcounter, 6))
xerror = np.zeros((endcounter, 6))
ydata = np.zeros((endcounter, 4))
yerror = np.zeros((endcounter, 4))

while counter < endcounter:

#%% - Load and Plot Image

    # load image
    image = Image.open("OA{}_24hr240313.tiff".format(counter+1))
    # take mean across layers
    image = np.mean(image, axis=2)
    image = np.transpose(image)
    
    # Normalise intesnity
    image = image / image.max()    
    
    # - Scale and Recenter
    
    # new x and y arrays which are scaled by the size of 1 pixel and centered on 0
    yr = np.linspace(-image.shape[0]/2, image.shape[0]/2, image.shape[0]) * pixel_size
    xr = np.linspace(-image.shape[1]/2, image.shape[1]/2, image.shape[1]) * pixel_size
    
    # # Check scaling of image
    # X, Y = np.meshgrid(xr, yr)
    # # Check the plot
    # plt.figure()
    # c1 = plt.contourf(X, Y, image, 100, cmap=cm.magma)
    # plt.xlabel("x (um)")
    # plt.ylabel("y (um)")
    # plt.title("Original Image Re-centered and scaled to Pixel Length")
    # cbar1 = plt.colorbar(c1)
    # cbar1.set_label("Intensity")
    
    
    
    #%% - Slicing

    # Sum along each axis to take mean results    
    Row = np.sum(image, axis=0)
    Column = np.sum(image, axis=1)
    
    # begin plot along x
    plt.figure()
    plt.plot(xr, Row)
    plt.xlabel("x (um)")
    plt.ylabel("Intensity ()")
    plt.title("Sum of Intenisty along y axis {}".format(counter+1))
    
    # Estimate fit parameters and fit along x
    
    k_g = 2*np.pi/100#  read from graph as distance between two peak 
                        # (sin^2 therefore 2 peaks per k)
    phi_g = 0
    sx_g = 100
    b_g = 3e4
    A_g = float(Row.max()) - b_g
    x0_g = -100 
    
    
    # fitting model takes parameters in following order
    p0x = [A_g, k_g, phi_g, x0_g, sx_g, b_g]
    
    
    # Attempt fit
    try:
        # pass to scipy fit function
        xfitvals, xerrs = opt.curve_fit(lf.sinefit, xr, Row, p0x)
        # (A_fx, k_fx, phi_fx, x0_fx, sx_fx, b_fx)

        # calculate standard deviation
        stdex = np.sqrt(np.diagonal(xerrs))

        xfitdata = lf.sinefit(xr, xfitvals[0], xfitvals[1], xfitvals[2], xfitvals[3], 
                              xfitvals[4], xfitvals[5])
        
        plt.plot(xr, xfitdata, 'r-')
        plt.legend(["Data Slice", "Fit"], loc='best')
        
        # update hold array
        xdata[counter, :] = xfitvals
        xerror[counter, :] = stdex
        
    except(RuntimeError):
        print('Iteration:', counter+1)
        xdata[counter, :] = np.zeros((1, 6))
        xerror[counter, :] = np.zeros((1, 6))

    
    
    # repeat along y
    plt.figure()
    plt.plot(yr, Column)
    plt.xlabel("y (pixel)")
    plt.ylabel("Intensity ()")
    plt.title("Slice through central fringe{}".format(counter+1))
    
    y0_g = -44
    sy_g = 100
    b_g = 3e4
    A_g = Column.max() - b_g
    p0y = [A_g, y0_g, sy_g, b_g]
    
    try:
        yfitvals, yerrs = opt.curve_fit(lf.expfit, yr, Column, p0y)
        # (A_fy, y0_fy, sy_fy, b_fy)

        stdey = np.sqrt(np.diag(yerrs))
        
        yfitdata = lf.expfit(yr, yfitvals[0], yfitvals[1], yfitvals[2], yfitvals[3])
        plt.plot(yr, yfitdata, 'r-')
        plt.legend(["Data Slice", "Fit"], loc='best')
        
        ydata[counter, :] = yfitvals
        yerror[counter, :] = stdey
        
    except(RuntimeError):

        print('Iteration:', counter+1)
        ydata[counter, :] = np.zeros((1, 4))
        yerror[counter, :] = np.zeros((1, 4))

    
    counter += 1
    
    
np.savetxt('xdata.txt', xdata, delimiter=',')
np.savetxt('ydata.txt', ydata, delimiter=',')
np.savetxt('xerror.txt', xerror, delimiter=',')
np.savetxt('yerror.txt', yerror, delimiter=',')
    

    


#%% - 2D Fitting

  # # Create array to hold all guess parameters
  # # [A,   k,   phi, x0,  y0, sigmax, sigmay, off]
  # passparams = np.array([A_fx, k_fx, phi_fx, x0_fx, y0_fy, sx_fx, sy_fy, b_fx])
 
  # #create empty hold array (test for iterative camera run)
  # fits = np.zeros((counter, 8))
  # # call to the 3D fitting function
  # img, resu = lf.fitimagesin(image, pixel_size, title_str, passparams)
  # # (Meshgrid2D_x,Meshgrid2D_y),fit_result=lf.fitimagesin(image, pixel_size, title_str, passparams)
 
  # print(img.fit_report())
  # # print('aspect ratio = (y/x)')
  # # print(fit_result.best_values['Centre_Gauss1_sigma_y']/fit_result.best_values['Centre_Gauss1_sigma_x'])
 
  # # # recover dictionary of fitted parameters
  # # dit = img[2]
  # # # recover variables only
  # # values = list(dit.values())
  # # # write to hold array
  # # fits[0, :] = values
  # # print(values)
    
    
