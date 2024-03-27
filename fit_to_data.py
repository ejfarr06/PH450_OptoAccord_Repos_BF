#!/usr/bin/env python
# -*- coding: utf-8 -*-

# An example of the programme used to fit to the data taken from the lab
# using the run from 19-20/03/24

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from PIL import Image
import fittingfuncs as ff

plt.close("all")

# 144 IMAGES

# FAILS AT 29, 32, 49, 56, 60
# Bad Fits at 27, 48, 49

# font sizes
title_font = 44
label_font = 40

#define parameters for camera run
stepmins = 10 # how many secs between images
timestep = 60 * stepmins - 2.8#timestep in seconds
endcounter = 143 # int(runtime / timestep) # total number of images
counter = 0
delay = 0.8

pixel_size = 1.55

# create empty array to hold fitting values
xdata = np.zeros((endcounter, 6))
xerror = np.zeros((endcounter, 6))
ydata = np.zeros((endcounter, 4))
yerror = np.zeros((endcounter, 4))

# create array of fail points (after first run)
fail = np.array([29, 32, 49, 56, 60])

# start run loop
while counter < endcounter:

                            
    #open the frame
    image0 = Image.open("PiRunImages/OA{}_24hr240319.tiff".format(counter+1))
    # average any possible layers in the image
    image = np.mean(image0, axis=2)
    image = np.transpose(image)
    # normalise
    image = image / image.max()
    
     # - Scale and Recenter

    # new x and y arrays which are scaled by the size of 1 pixel and centered on 0
    yr = np.linspace(-image.shape[0]/2, image.shape[0]/2, image.shape[0]) * pixel_size
    xr = np.linspace(-image.shape[1]/2, image.shape[1]/2, image.shape[1]) * pixel_size
    
    
    # Slicing

    # Sum along each axis to take mean results    
    Row = np.sum(image, axis=0)
    Column = np.sum(image, axis=1)
    
    # Estimate fit parameters and fit along x
    
    k_g = 2*np.pi/100 #  read from graph as no. pixels between peaks
    phi_g = 0
    sx_g = 100
    b_g = 4e4
    A_g = float(Row.max()) - b_g
    x0_g = -100 
    
    
    # fitting model takes parameters in following order
    p0x = [A_g, k_g, phi_g, x0_g, sx_g, b_g]
    
    try:
        # pass to scipy fit function
        xfitvals, xerrs = opt.curve_fit(ff.sinefit, xr, Row, p0x, maxfev = 3000)
        # (A_fx, k_fx, phi_fx, x0_fx, sx_fx, b_fx)
        
        # calculate standard deviation
        stdex = np.sqrt(np.diagonal(xerrs))
        
        xdata[counter, :] = xfitvals
        xerror[counter, :] = stdex
        
        # Plot fit data
        xfitdata = ff.sinefit(xr, xfitvals[0], xfitvals[1], xfitvals[2], xfitvals[3], 
                              xfitvals[4], xfitvals[5])
        
        
        # if counter % 30 == 0:
        #     # begin plot along x
        #     plt.figure()
        #     plt.plot(xr, Row)
        #     plt.xlabel("x (um)", fontsize=label_font)
        #     plt.ylabel("Intensity ()", fontsize=label_font)
        #     plt.title("Summed Intensity On X Axis, Image {}".format(counter+1), fontsize=title_font)
        #     plt.plot(xr, xfitdata, 'r-')
        #     plt.legend(["Data Slice", "Fit"], loc='best')
            
    except:
        print("Fail x, ", counter)
        xdata[counter, :] = xdata[counter-1, :]
        xerror[counter, :] = xerror[counter-1, :]
        
    
    # repeat along y
    
    y0_g = -44
    sy_g = 100
    b_g = 3e4
    A_g = Column.max() - b_g
    p0y = [A_g, y0_g, sy_g, b_g]
    
    try:
        yfitvals, yerrs = opt.curve_fit(ff.expfit, yr, Column, p0y)
        
#         # (A_fy, y0_fy, sy_fy, b_fy)
#         
        stdey = np.sqrt(np.diag(yerrs))
        
        ydata[counter, :] = yfitvals
        yerror[counter, :] = stdey
        
        yfitdata = ff.expfit(yr, yfitvals[0], yfitvals[1], yfitvals[2], yfitvals[3])
        
        # if counter % 30 == 0:
        #     plt.figure()
        #     plt.plot(yr, Column)
        #     plt.xlabel("y (pixel)", fontsize=label_font)
        #     plt.ylabel("Intensity ()", fontsize=label_font)
        #     plt.title("Summed Intensity on Y Axis, Image {}".format(counter+1), fontsize=title_font)
        #     plt.plot(yr, yfitdata, 'r-')
        #     plt.legend(["Data Slice", "Fit"], loc='best')
        #     # plt.savefig("YFitGraph{}".format(counter+1))
        #     plt.show()
    except:
        print("Fail y, ", counter)
        ydata[counter, :] = ydata[counter-1, :]
        yerror[counter, :] = yerror[counter-1, :]
    
    if counter == 27 or counter == 48 or counter == 49 or counter == 89:
        xdata[counter, :] = xdata[counter-1, :]
        xerror[counter, :] = xerror[counter-1, :]
        ydata[counter, :] = ydata[counter-1, :]
        yerror[counter, :] = yerror[counter-1, :]
    
    counter += 1
       

                        



# np.savetxt("xdata_wout_fail.txt", xdata, delimiter=',')
# np.savetxt("ydata_wout_fail.txt", ydata, delimiter=',')
# np.savetxt("xerror_wout_fail.txt", xerror, delimiter=',')
# np.savetxt("yerror_wout_fail.txt", yerror, delimiter=',')

xfails = np.array([xdata[27], xdata[29], xdata[32], xdata[48], xdata[49], xdata[56], xdata[60], xdata[89], xdata[114]])
xerrfl = np.array([xerror[27], xerror[29], xerror[32], xerror[48], xerror[49], xerror[56], xerror[60], xerror[89], xerror[114]])
yfails = np.array([ydata[27], ydata[29], ydata[32], ydata[48], ydata[49], ydata[56], ydata[60], ydata[89], ydata[114]])

#%% Plot in Time
t = np.arange(0, (stepmins*60-delay)*endcounter, (stepmins*60-delay)) / 60
tifail = np.array([t[27], t[29], t[32], t[48], t[49], t[56], t[60], t[89], t[114]])

waveno = xdata[:, 1]
wavefail = xfails[:, 1]
# sin^2 fit therefore 2 peaks per wavelength, lattice peridon is wavelength/2
period = 2 * np.pi / (2*waveno)
pfail = 2 * np.pi / (2*wavefail)


fig, axs = plt.subplots(2, 3, layout='constrained')
fig.suptitle("X Fit Results (Across Lattice), 24hr Attempt 19-20/03/24", fontsize=40)
plt.tick_params(axis='both', labelsize=30)


axs[0, 0].errorbar(t, xdata[:, 0], xerror[:, 0], linestyle='-', marker='o', color='0.0')
axs[0, 0].plot(tifail, xfails[:, 0], 'ro', markersize=10)
# axs[0, 0].set_xlabel("Time (mins)")
axs[0, 0].set_ylabel("Peak Intensity", fontsize=label_font)
axs[0, 0].tick_params(axis='both', labelsize=20)


axs[0, 1].errorbar(t, period, xerror[:, 1], linestyle='-', marker='o', color='0.0')
axs[0, 1].plot(tifail, pfail, 'ro', markersize=10)
# axs[0, 1].set_xlabel("Time (mins)")
axs[0, 1].set_ylabel("Space. a (um)", fontsize=label_font)
axs[0, 1].tick_params(axis='both', labelsize=20)


axs[0, 2].errorbar(t, xdata[:, 2], xerror[:, 2], linestyle='-', marker='o', color='0.0')
# axs[0, 2].set_xlabel("Time (mins)")
axs[0, 2].set_ylabel("Phase (rads)", fontsize=label_font)
axs[0, 2].tick_params(axis='both', labelsize=20)
axs[0, 2].plot(tifail, xfails[:, 2], 'ro', markersize=10)

axs[1, 0].errorbar(t, xdata[:, 3], xerror[:, 3], linestyle='-', marker='o', color='0.0')
axs[1, 0].set_xlabel("Time (mins)", fontsize=label_font)
axs[1, 0].set_ylabel("Cent. x0 (um)", fontsize=label_font)
axs[1, 0].tick_params(axis='both', labelsize=20)
axs[1, 0].plot(tifail, xfails[:, 3], 'ro', markersize=10)

axs[1, 1].errorbar(t, xdata[:, 4], xerror[:, 4], linestyle='-', marker='o', color='0.0')
axs[1, 1].set_xlabel("Time (mins)", fontsize=label_font)
axs[1, 1].set_ylabel("Var(x) (um^2)", fontsize=label_font)
axs[1, 1].tick_params(axis='both', labelsize=20)
axs[1, 1].plot(tifail, xfails[:, 4], 'ro', markersize=10)

axs[1, 2].errorbar(t, xdata[:, 5], xerror[:, 5], linestyle='-', marker='o', color='0.0')
axs[1, 2].set_xlabel("Time (mins)", fontsize=label_font)
axs[1, 2].set_ylabel("Int. Offset", fontsize=label_font)
axs[1, 2].tick_params(axis='both', labelsize=20)
axs[1, 2].plot(tifail, xfails[:, 5], 'ro', markersize=10)

plt.show()




fig, axs = plt.subplots(2, 2, layout='constrained')
fig.suptitle("Y Fit Results (Across Fringe), 24hr Attempt 19-20/03/24", fontsize=title_font)

axs[0, 0].errorbar(t, ydata[:, 0], yerror[:, 0], linestyle='-', marker='o', color='C0')
# axs[0, 0].set_xlabel("Time (mins)", fontsize=label_font)
axs[0, 0].set_ylabel("Peak Int.", fontsize=label_font)
axs[0, 0].tick_params(axis='both', labelsize=20)
axs[0, 0].plot(tifail, yfails[:, 0], 'ro', markersize=10)

axs[0, 1].errorbar(t, ydata[:, 1], yerror[:, 1], linestyle='-', marker='o', color='C0')
# axs[0, 1].set_xlabel("Time (mins)", fontsize=label_font)
axs[0, 1].set_ylabel("Cent. y0 (um)", fontsize=label_font)
axs[0, 1].tick_params(axis='both', labelsize=20)
axs[0, 1].plot(tifail, yfails[:, 1], 'ro', markersize=10)

axs[1, 0].errorbar(t, ydata[:, 2], yerror[:, 2], linestyle='-', marker='o', color='C0')
axs[1, 0].set_xlabel("Time (mins)", fontsize=label_font)
axs[1, 0].set_ylabel("Var(y) (um^2)", fontsize=label_font)
axs[1, 0].tick_params(axis='both', labelsize=20)
axs[1, 0].plot(tifail, yfails[:, 2], 'ro', markersize=10)

axs[1, 1].errorbar(t, ydata[:, 3], yerror[:, 3], linestyle='-', marker='o', color='C0')
axs[1, 1].set_xlabel("Time (mins)", fontsize=label_font)
axs[1, 1].set_ylabel("Int. Offset", fontsize=label_font)
axs[1, 1].tick_params(axis='both', labelsize=20)
axs[1, 1].plot(tifail, yfails[:, 3], 'ro', markersize=10)














