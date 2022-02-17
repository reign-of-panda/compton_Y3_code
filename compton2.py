# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 12:05:25 2022

@author: therm
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import os
from scipy.stats import chisquare
import scipy.odr
import copy

plt.rcParams['font.size'] = 12.5

"""
Extract the data first
"""

def txt_unpack(f_name):
    """
    This 
    """
    path = os.getcwd()
    all_files = os.listdir(path)
    # print(all_files)
        
    data = pd.read_csv(f_name, header = None, skiprows = 5, delimiter = '\\t', engine = 'python')
    data = data.select_dtypes([np.number])
    number_of_cols = len(data.columns) # Because first two columns are nan
    
    # Relabel headers
    old_headers = data.columns
    new_headers = np.arange(0, number_of_cols, 1, dtype = 'int')
    header_dict = dict(zip(old_headers, new_headers))
    data.rename(columns = header_dict, inplace = True)
    
    # Energy array
    x = np.arange(1, 513, 1)
    return x, data

def gauss(x, a, b, c):
    return a * np.exp(-((x+b)**2)/(2*c**2))

def least_squares(x_data, y_data, fit_param, fit_func):
    y_fit = fit_func(x_data, *fit_param)
    lsq = (y_data - y_fit) ** 2
    return lsq
    
def fit_gauss(x_lims, E_arr, count_arr, count_err):
    lim1 = x_lims[0]
    lim2 = x_lims[1]
    
    mask = (E_arr >= lim1) & (E_arr <= lim2)
    E_arr = E_arr[mask]
    count_arr = count_arr[mask]
    count_err = count_err[mask]
    
    # Compute a guess for 
    argument = np.argmax(abs(count_arr))
    a = count_arr[argument] #max(abs(count_arr))
    b = (lim2 + lim1)/2
    c = (lim2 - lim1)
    guess = [a, b, c]
        
    par, cov = curve_fit(gauss, E_arr, count_arr, p0 = guess, sigma = count_err, absolute_sigma = True)
    b_err = np.sqrt(cov[1][1])
    
    x_fit = np.arange(0, 512, 1) #np.linspace(lim1, lim2, 100)
    y_fit = gauss(x_fit, *par)
    
    # Calculate least squares
    lsq_arr = least_squares(E_arr, count_arr, par, gauss)
    
    b_err *= np.std(np.sqrt(lsq_arr)) 
    
    # Peak energy
    central_energy = -par[1]
    return x_fit, y_fit, central_energy, lsq_arr, b_err

def var_lim(lim, x_data, y_data, y_err):
    """
    Enter max lim, and then reduce slowly to see where you get the smallest least squares
    """
    lim1 = lim[0]
    lim2 = lim[1]
    del_lim = 1
    all_peak = []
    
    min_lsq = -1 
    best_lim = lim
    for i in range(14): # The range choice here is arbitrary
        try:
            x_fit, y_fit, peak_E, lsq, b_err = fit_gauss(lim, x_data, y_data, y_err) 
        except:
            x_fit, y_fit, peak_E, lsq, b_err = fit_gauss(best_lim, x_data, y_data, y_err) 
        lsq = sum(lsq)
        if (min_lsq == -1) or (lsq < min_lsq):
            min_lsq = lsq
            best_lim = lim
    
        lim1 += del_lim
        lim2 -= del_lim
        lim = [lim1, lim2]
        
        all_peak.append(peak_E)
        
    # print("VARLIM OUT")
    # print("std:", np.std(all_peak))
        
    return best_lim

# Calibration stuff
def lin_odr(C, x):
    return C[0] * x + C[1]

# def comp_odr(C, x):
#     return C[0] / (1 + C[1] * (1 - np.cos(x)))

def compton_curve(x, a, b):
    return a / (1 + b * (1 - np.cos(x)))

def ODR_linear(V, A, V_err):
    """
    Takes care of linear fitting
    """
    # Use curve_fit to generate guesses
    params1, cov1 = curve_fit(linear, V, A)

    model_Y = scipy.odr.Model(lin_odr)
    mydata_Y = scipy.odr.RealData(V, A, sx=V_err)
    myodr_Y = scipy.odr.ODR(mydata_Y, model_Y, beta0=params1)
    myoutput_Y = myodr_Y.run()
    
    parameters = myoutput_Y.beta
    param_error = myoutput_Y.sd_beta
    
    myoutput_Y.pprint()

    return parameters, param_error

# Measured to true energy conversion
def energy_converter(E, E_err, par, err):
    """
    E is the measured energy array
    ONLY ENTER NUMPY ARRAYS FOR ANY MULTIDIMENSIONAL INPUTS (except the parameters)
    
    mE_c: m*E + c
    m_E: m*E
    """
    m = par[0]
    c = par[1]
    m_err = err[0]
    c_err = err[1]
    
    m_E = m * E
    relative_err = E_err/E + m_err/m
    m_E_err = m_E * relative_err
    
    mE_c = m_E + c
    mE_c_err = np.sqrt(m_E_err**2 + c_err**2)
    
    return mE_c, mE_c_err

#%%
"""
Perform all the data unpacking here
"""
# Headers to use 
even_head = [0, 2, 4, 6, 8, 10, 12, 14, 16]
odd_head = [3, 5, 7, 9, 11, 13, 15]
all_heads = np.sort(even_head + odd_head)

# This is the no target data (even anglular intervals)
x1, dF1 = txt_unpack("Day5_Cs137_0x_14x_noTarget.txt")
dF1 = dF1.drop(columns = 8) # Gets rid of empty column
x2, dF2 = txt_unpack("Day5_Cs137_16x_noTarget.txt")
dF_noTar_even = pd.concat([dF1, dF2], axis = 1, ignore_index=True)
dF_noTar_even.columns = even_head

# Data with target (even anglular intervals)
# These files contain background columns - remove them
this_sucks = ['0', '2x', '4x', '6x', '8x', '10x', '12x', '14x', '16x']
day_even = 4
all_dF = []
for i in this_sucks:
    x, dF = txt_unpack('Day{}_Cs137_{}deg.txt'.format(day_even, i))
    if np.shape(dF)[1] == 2: # if statements get rid of background data columns
        source_data = dF[1]
    elif np.shape(dF)[1] == 1:
        source_data = dF[0]
    all_dF.append(source_data)

dF_even = pd.concat(all_dF, axis = 1, ignore_index=True)
dF_even.columns = even_head

# Data with no target (odd angular intervals)
this_sucks2 = ['3x', '5x', '7x', '9x', '11x', '13x', '15x']
x_odd, dF_noTar_odd = txt_unpack("Day6_Cs137_3x_15xdeg_noTarget.txt")
dF_noTar_odd.columns = odd_head

# Data with target (odd anglular intervals)
x_odd, dF_odd = txt_unpack("Day6_Cs137_3x_15xdeg.txt")
dF_odd.columns = odd_head

# Now combine the odd and even data
dF_tot = pd.concat([dF_even, dF_odd], axis = 1)
dF_tot = dF_tot[all_heads]

dF_tot_noTar = pd.concat([dF_noTar_even, dF_noTar_odd], axis = 1)
dF_tot_noTar = dF_tot_noTar[all_heads] # This reorders the columns by heading

# Differential 
dF_diff = dF_tot - dF_tot_noTar
dF_diff[0] = dF_tot[0] 
dF_diff[2] = dF_tot[2] 
dF_diff[3] = dF_tot[3]

dF_diff_err = np.sqrt(dF_tot + dF_tot_noTar) # + dF_tot_noTar
dF_diff_err[0] = np.sqrt(dF_tot[0])
dF_diff_err[2] = np.sqrt(dF_tot[2])
dF_diff_err[3] = np.sqrt(dF_tot[3])

x = x1

#%%
# Fit gaussians
# limit_dict = dict([
#             (0, [160, 250]),
#             (2, [180, 240]),
#             (3, [190, 260]),
#             (4, [135, 240]),
#             (5, [120, 220]),
#             (6, [120, 220]),
#             (7, [100, 180]),
#             (8, [100, 180]),
#             (9, [100, 160]),
#             (10, [90, 160]),
#             (11, [70, 160]),
#             (12, [60, 120]),
#             (13, [60, 140]),
#             (14, [70, 140]),
#             (15, [60, 120]),
#             (16, [60, 120]),
#             ])

# Not using differntial spectrum for 0x, 2x, and 3x
limit_dict = dict([
            (0, [160, 250]),
            (2, [180, 240]),
            (3, [175, 240]),
            (4, [135, 240]),
            (5, [120, 220]),
            (6, [120, 220]),
            (7, [100, 180]),
            (8, [100, 180]),
            (9, [100, 160]),
            (10, [90, 160]),
            (11, [70, 160]),
            (12, [60, 120]),
            (13, [60, 140]),
            (14, [70, 140]),
            (15, [60, 120]),
            (16, [60, 120]),
            ])
ind = 3

M_energy = []
M_energy_err = []
fits = []
for i in all_heads:
    print(i)
    count_data = np.array(dF_diff[i])
    count_err = np.array(dF_diff_err[i])
    lim_guess = limit_dict[i]
    print(lim_guess)
    new_lim = var_lim(lim_guess, x, count_data, count_err)
    x_fit, y_fit, central_energy, lsq_arr, b_err = fit_gauss(new_lim, x, count_data, count_err)
    M_energy.append(central_energy)
    M_energy_err.append(b_err)
    fits.append([x_fit, y_fit])
    
fit_dict = dict(zip(all_heads,fits))
M_energy = np.array(M_energy)
M_energy_err = np.array(M_energy_err)

# Plot 
for i in all_heads:
    plt.bar(x, dF_diff[i],  width = 1, label = str(i) + 'x')
    temp_fit = fit_dict[i]
    fitx = temp_fit[0]
    fity = temp_fit[1]
    plt.plot(fitx, fity, label = "Fit", color = 'black')
    plt.xlabel("Measured Energy (keV)")
    plt.ylabel("Count")
    plt.grid(which = 'minor', alpha = 0.2)
    plt.grid(which = 'major', alpha = 0.6)
    plt.minorticks_on()
    plt.legend()
    plt.show()


# Calibration params
calibrate_params = [3.49367515, -30.10126123]
calibrate_err = [0.01449711, 1.87941578]

ang_x = 5.625 * np.pi/180 # This is in rads
angles = np.array(all_heads) * ang_x 
angle_err = np.ones(len(angles)) * 0.35 * np.pi/180 # Also in rads

true_E_comp, true_E_err = energy_converter(M_energy, M_energy_err, calibrate_params, calibrate_err)

# The theoretical Compton effect curve
b_par = 662000*1.6e-19 / (9.11e-31*(3e8)**2)
par2 = [662,b_par]

x_arr = np.linspace(angles[0], angles[-1], 100)
y_arr = compton_curve(x_arr, *par2)

# Plots
plt.plot(x_arr, y_arr, label = "True")
plt.errorbar(angles, true_E_comp, yerr = true_E_err, xerr = angle_err, fmt = '.', label = "True Energy")
plt.xlabel('Angle (rad)')
plt.ylabel('Peak Energy (keV)')
plt.grid(which = 'minor', alpha = 0.2)
plt.grid(which = 'major', alpha = 0.6)
plt.minorticks_on()
plt.legend()
plt.show()

#%%
"""
Chi squared 
"""
x_arr = np.linspace(angles[0], angles[-1], 16)
y_arr = compton_curve(x_arr, *par2)

chi_sq, p_val = chisquare(true_E_comp, y_arr)

print(chi_sq, p_val)

#%%
"""
Save the data
"""
# data_save = np.array([angles, true_E_comp, angle_err, true_E_err]).T
# dF_save = pd.DataFrame(data_save, columns = ['angles', 'trueE', 'angle_err', 'trueE_err'])
# dF_save.to_csv('FINAL_COMPTON_EFFECT.csv')



















