# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 09:43:32 2022

@author: therm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import os
from scipy.stats import chisquare
plt.rcParams['font.size'] = 12.5

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
    
    mask = (E_arr >= lim1) & (E_arr < lim2)
    E_arr = E_arr[mask]
    count_arr = count_arr[mask]
    count_err = count_err[mask]
    
    # Compute a guess for curve_fit
    a = max(count_arr)
    b = (lim2 + lim1)/2
    c = (lim2 - lim1)
    guess = [a, b, c]
        
    par, cov = curve_fit(gauss, E_arr, count_arr, p0 = guess, sigma = count_err, absolute_sigma = True)
    b_err = np.sqrt(cov[1][1])
    
    x_fit = np.arange(0, 512, 1) #np.linspace(lim1, lim2, 100)
    y_fit = gauss(x_fit, *par)
    
    # Calculate least squares
    lsq_arr = least_squares(E_arr, count_arr, par, gauss)
    
    # Peak energy
    central_energy = -par[1]
    return x_fit, y_fit, central_energy, lsq_arr, b_err
    

def dict_conv(x):
    """
    x is the dictionary
    """
    temp = np.array(list(x.items())).T[1]
    temp = np.array(temp, dtype = 'float')

    return temp  

def linear(x, m, c):
    return m*x + c



#%%
"""
Compton effect investigation
"""

limit_dict = dict([
            ('0', [160, 250]),
            ('2x', [160, 250]),
            ('4x', [135, 240]),
            ('6x', [120, 220]),
            ('8x', [100, 180]),
            ('10x', [100, 160]),
            ('12x', [70, 145]),
            ('14x', [60, 140]),
            ('16x', [60, 120]),
            ])


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

def background_mean():
    """
    Averaging the backgrounds from certain files
    """
    day = 4
    pre_deg = ['0', '2x', '6x', '8x', '10x']
    
    background = 0
    for i in pre_deg:
        x, dF = txt_unpack('Day{}_Cs137_{}deg.txt'.format(day, i))
        background += dF[0]
    
    mean_back = background / 5    
    return mean_back
    

def plot_figure(day, pre_deg):
    """
    day and pre_deg specify file name
    They are strings
    """
    x, dF = txt_unpack('Day{}_Cs137_{}deg.txt'.format(day, pre_deg))
    
    # This if statement deals with the fact that some of the data files don't have a 
    # background data column.
    if np.shape(dF)[1] == 2:
        source_data = dF[1]
    elif np.shape(dF)[1] == 1:
        source_data = dF[0]
    
    # signal - background
    back_remove = True
    background = background_mean()
    if back_remove:
        sig_min_back = np.array((source_data - background))
        sig_min_back = np.clip(sig_min_back, a_min = 0, a_max = None) # Clip to remove -ve values
        sig_min_back_err = np.sqrt(sig_min_back + background)
    else:
        sig_min_back = source_data
        sig_min_back_err = np.sqrt(sig_min_back)
    
    # Curve fit
    x_lims = limit_dict[pre_deg]
    x_lims = var_lim(x_lims, x, sig_min_back, sig_min_back_err) # Best limit
    print("Best lim:", x_lims)
    x_fit, y_fit, peak_E, lsq, b_err = fit_gauss(x_lims, x, sig_min_back, sig_min_back_err) 
    
    # Plotting
    plt.bar(x, sig_min_back, width = 1, label = "Cs-137, $\\theta = {}\degree$".format(pre_deg), align = 'center')
    plt.plot(x_fit, y_fit, color = 'black')
    plt.xlabel("Measured Energy (keV)")
    plt.ylabel("Count")
    plt.grid(which = 'minor', alpha = 0.2)
    plt.grid(which = 'major', alpha = 0.6)
    plt.minorticks_on()
    plt.legend()
    plt.show()
    
    return peak_E , b_err # b_err is the error on peak_E


#%%
"""
Day2_calibration_700V.txt

N1, N2, N3: Cs-137, Co-57, Am-241
"""
x, dF = txt_unpack('Day2_calibration_700V.txt')
measured_peak_E_arr = []
measured_peak_E_err = []

# Cs-137
lim_guess = [170, 240]
x_lims = var_lim(lim_guess, x, dF[0], np.sqrt(dF[0]))
x_fit, y_fit, peak_E, lsq, b_err = fit_gauss(x_lims, x, dF[0], np.sqrt(dF[0])) 
measured_peak_E_arr.append(peak_E)
measured_peak_E_err.append(b_err)

plt.plot(x_fit, y_fit, color = 'black')
plt.bar(x, dF[0], width = 1, label = "Cs-137 Calibrate", align = 'center')
plt.xlabel("Energy (keV)")
plt.ylabel("Count")
plt.grid(which = 'minor', alpha = 0.2)
plt.grid(which = 'major')
plt.minorticks_on()
plt.legend()
plt.show()

# Co-57
lim_guess = [20, 60]
x_lims = var_lim(lim_guess, x, dF[1], np.sqrt(dF[1]))
x_fit, y_fit, peak_E, lsq, b_err = fit_gauss(x_lims, x, dF[1], np.sqrt(dF[1])) 
measured_peak_E_arr.append(peak_E)
measured_peak_E_err.append(b_err)

plt.plot(x_fit, y_fit, color = 'black')
plt.bar(x, dF[1], width = 1, label = "Co-57 Calibrate", align = 'center')
plt.xlabel("Energy (keV)")
plt.ylabel("Count")
plt.grid(which = 'minor', alpha = 0.2)
plt.grid(which = 'major')
plt.minorticks_on()
plt.legend()
plt.show()

# Am-241
lim_guess = [15, 30]
x_lims = var_lim(lim_guess, x, dF[2], np.sqrt(dF[2]))
x_fit, y_fit, peak_E, lsq, b_err = fit_gauss(x_lims, x, dF[2], np.sqrt(dF[2])) 
measured_peak_E_arr.append(peak_E)
measured_peak_E_err.append(b_err)

plt.plot(x_fit, y_fit, color = 'black')
plt.bar(x, dF[2], width = 1, label = "Am-241 Calibrate", align = 'center')
plt.xlabel("Energy (keV)")
plt.ylabel("Count")
plt.grid(which = 'minor', alpha = 0.2)
plt.grid(which = 'major')
plt.minorticks_on()
plt.xlim(0, 140)
plt.legend()
plt.show()

x, dF = txt_unpack('Day2_calibration_700V_run2.txt')

# Cd-109
lim_guess = [25, 40]
counts = dF[0]
x_lims = var_lim(lim_guess, x, counts, np.sqrt(counts))
x_fit, y_fit, peak_E, lsq, b_err = fit_gauss(x_lims, x, counts, np.sqrt(counts)) 
measured_peak_E_arr.append(peak_E)
measured_peak_E_err.append(b_err)

plt.plot(x_fit, y_fit, color = 'black')
plt.bar(x, counts, width = 1, label = "Cd-109 Calibrate", align = 'center')
plt.xlabel("Energy (keV)")
plt.ylabel("Count")
plt.grid(which = 'minor', alpha = 0.2)
plt.grid(which = 'major')
plt.minorticks_on()
plt.legend()
plt.xlim(0, 100)
plt.show()

# Na-22
lim_guess = [100, 200]
counts = dF[1]
x_lims = var_lim(lim_guess, x, counts, np.sqrt(counts))
x_fit, y_fit, peak_E, lsq, b_err = fit_gauss(x_lims, x, counts, np.sqrt(counts)) 
measured_peak_E_arr.append(peak_E)
measured_peak_E_err.append(b_err)

plt.plot(x_fit, y_fit, color = 'black')
plt.bar(x, counts, width = 1, label = "Na-22 Calibrate", align = 'center')
plt.xlabel("Energy (keV)")
plt.ylabel("Count")
plt.grid(which = 'minor', alpha = 0.2)
plt.grid(which = 'major')
plt.minorticks_on()
plt.legend()
# plt.xlim(0, 100)
plt.show()

# Mn-54
lim_guess = [200, 280]
counts = dF[2]
x_lims = var_lim(lim_guess, x, counts, np.sqrt(counts))
x_fit, y_fit, peak_E, lsq, b_err = fit_gauss(x_lims, x, counts, np.sqrt(counts)) 
measured_peak_E_arr.append(peak_E)
measured_peak_E_err.append(b_err)

plt.plot(x_fit, y_fit, color = 'black')
plt.bar(x, counts, width = 1, label = "Mn-54 Calibrate", align = 'center')
plt.xlabel("Energy (keV)")
plt.ylabel("Count")
plt.grid(which = 'minor', alpha = 0.2)
plt.grid(which = 'major')
plt.minorticks_on()
plt.legend()
# plt.xlim(0, 100)
plt.show()

# Co-60-P1 and Co-60-P2
# THIS DOESN'T WORK WITH CURVE_FIT

# lim_guess = [325, 355]
# counts = dF[3]
# null_err = np.zeros(len(counts))
# x_lims = var_lim(lim_guess, x, counts, np.sqrt(null_err))
# x_fit, y_fit, peak_E, lsq, b_err = fit_gauss(lim_guess, x, counts, null_err) 
# measured_peak_E_arr.append(peak_E)
# measured_peak_E_err.append(b_err)

# plt.plot(x_fit, y_fit, color = 'black', label = "Peak 1")

# lim_guess = [360, 420]
# counts = dF[3]
# x_lims = var_lim(lim_guess, x, counts, np.sqrt(counts))
# x_fit, y_fit, peak_E, lsq, b_err = fit_gauss(x_lims, x, counts, np.sqrt(counts)) 
# measured_peak_E_arr.append(peak_E)
# measured_peak_E_err.append(b_err)

# plt.plot(x_fit, y_fit, color = 'black', label = "Peak 2")


# plt.bar(x, counts, width = 1, label = "Co-60 Calibrate", align = 'center')
# plt.xlabel("Energy (keV)")
# plt.ylabel("Count")
# plt.grid(which = 'minor', alpha = 0.2)
# plt.grid(which = 'major')
# plt.minorticks_on()
# plt.legend()
# # plt.xlim(0, 100)
# plt.show()

# Ba-133
# lim_guess = [0, 18]
counts = dF[4]
# x_lims = var_lim(lim_guess, x, counts, np.sqrt(counts))
# x_fit, y_fit, peak_E, lsq, b_err = fit_gauss(lim_guess, x, counts, np.sqrt(counts)) 
# measured_peak_E_arr.append(peak_E)
# measured_peak_E_err.append(b_err)

# plt.plot(x_fit, y_fit, color = 'black', label = "Peak 1")

lim_guess = [25, 40]
x_lims = var_lim(lim_guess, x, counts, np.sqrt(counts))
x_fit, y_fit, peak_E, lsq, b_err = fit_gauss(x_lims, x, counts, np.sqrt(counts)) 
measured_peak_E_arr.append(peak_E)
measured_peak_E_err.append(b_err)

plt.plot(x_fit, y_fit, color = 'black', label = "Peak 2")

lim_guess = [100, 135]
x_lims = var_lim(lim_guess, x, counts, np.sqrt(counts))
x_fit, y_fit, peak_E, lsq, b_err = fit_gauss(x_lims, x, counts, np.sqrt(counts)) 
measured_peak_E_arr.append(peak_E)
measured_peak_E_err.append(b_err)
plt.plot(x_fit, y_fit, color = 'brown', label = "Peak 3")

plt.bar(x, counts, width = 1, label = "Co-60 Calibrate", align = 'center')
plt.xlabel("Energy (keV)")
plt.ylabel("Count")
plt.grid(which = 'minor', alpha = 0.2)
plt.grid(which = 'major')
plt.minorticks_on()
plt.legend()
plt.xlim(0, 140)
plt.show()

#%%
"""
Calibration
"""
# in keV
true_energy = dict([
            ('Cs-137', 662),
            ('Co-57', 122),
            ('Am-241', 59.6),
            ('Cd-109', 88),
            ('Na-22', 511),
            ('Mn-54', 834.8),
            # ('Co-60-P1', 1173.2),
            # ('Co-60-P2', 1332.5),
            # ('Ba-133-P1', 31),
            ('Ba-133-P2', 80),
            ('Ba-133-P3', 356)
            ])


# measured_E_700 = dict([
#             ('Cs-137', 198),
#             ('Co-57', 43),
#             ('Am-241', 23),
#             ('Cd-109', 32),
#             ('Na-22', 154),
#             ('Mn-54', 246), # This one's iffy
#             ('Co-60-P1', 342),
#             ('Co-60-P2', 384),
#             ('Ba-133-P1', 11),
#             ('Ba-133-P2', 31),
#             ('Ba-133-P3', 111)
#             ])

true_arr = np.array(dict_conv(true_energy))
measure_700_arr = np.array(measured_peak_E_arr)
measure_700_err = np.array(measured_peak_E_err)
# measure_700_arr = dict_conv(measured_E_700)
# factor = true_arr/measure_700_arr

# ODR fitting
import scipy.odr

def lin_odr(C, x):
    return C[0] * x + C[1]

def comp_odr(C, x):
    return C[0] / (1 + C[1] * (1 - np.cos(x)))

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

def ODR_compton(V, A, V_err, A_err):
    """
    Takes care of compton fitting
    """
    params1, cov1 = curve_fit(compton_curve, V, A, p0 = [700, 0.7])
    model_Y = scipy.odr.Model(comp_odr)
    mydata_Y = scipy.odr.RealData(V, A, sx=V_err, sy=A_err)
    myodr_Y = scipy.odr.ODR(mydata_Y, model_Y, beta0=params1)
    myoutput_Y = myodr_Y.run()
    
    parameters = myoutput_Y.beta
    param_error = myoutput_Y.sd_beta
    
    myoutput_Y.pprint()

    return parameters, param_error


calibrate_params, calibrate_err = ODR_linear(measure_700_arr, true_arr, measure_700_err)
x_arr = np.linspace(0, 280, 100)
fit = linear(x_arr, *calibrate_params)

plt.errorbar(measure_700_arr, true_arr, xerr = measure_700_err, fmt='o', label = "700V", capsize = 2)
plt.plot(x_arr, fit)
plt.grid(which = 'minor', alpha = 0.2)
plt.grid(which = 'major')
plt.minorticks_on()
plt.xlabel("Measured Energy (keV)")
plt.ylabel("True Energy (keV)")
plt.legend()
plt.show()

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
Investigating the Compton effect
"""
measured_peakE = []
measured_E_err = []
ang_x = 5.625
angles = np.arange(0, 18*ang_x, 2*ang_x) * np.pi/180
angle_err = np.ones(len(angles)) * 0.35 * np.pi/180 # Calculation in lab note book errors
this_sucks = ['0', '2x', '4x', '6x', '8x', '10x', '12x', '14x', '16x']
for i in this_sucks:
    peak_E, peak_E_err = plot_figure(4, i)
    measured_peakE.append(peak_E)
    measured_E_err.append(peak_E_err)
    
measured_peakE = np.array(measured_peakE)
measured_E_err = np.array(measured_E_err)
true_E_comp, true_E_err = energy_converter(measured_peakE, measured_E_err, calibrate_params, calibrate_err)
#%%

# par2, cov2 = curve_fit(compton_curve, angles, true_E_comp, p0 = [700, 0.7])
# print(par2)

par2, err2 = ODR_compton(angles, true_E_comp, angle_err, true_E_err)

x_arr = np.linspace(0, 18*ang_x, 100) * np.pi/180
y_arr = compton_curve(x_arr, *par2)

plt.plot(x_arr, y_arr, label = "Fit")
plt.errorbar(angles, true_E_comp, yerr = true_E_err, xerr = angle_err, fmt = '.', label = "True Energy")
plt.xlabel('Angle (rad)')
plt.ylabel('Peak Energy (keV)')
plt.grid(which = 'minor', alpha = 0.2)
plt.grid(which = 'major', alpha = 0.6)
plt.minorticks_on()
plt.legend()
plt.show()

chi_sq, p_val = chisquare(true_E_comp, compton_curve(angles, *par2))



























