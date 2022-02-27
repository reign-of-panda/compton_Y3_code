# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:30:22 2022

@author: therm
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import bisect
import math as mt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 13})

# np.random.seed(7)

class sig_sim:
    def __init__(self):
        # Geometry parameters
        self.targetD = 0.02 # Target diameter in meters
        self.sourceA = 0.008 # Source aperture diameter
        self.detectorA = 0.02 # Detector aperture diameter
        self.S_to_T_D = 16.4/100 # Distance from source to target (this probably won't be used)
        self.T_to_D_D = 12/100 # Distance from target to detector 
        
        # Number of photons
        self.N = int(5e6)
        
        # Target material properties
        self.Z = 13 # Aluminium atomic number
        self.n_Al = 2700/(27*1.67e-27) # Number density of Aluminium
        
        # Simulation details
        self.maximum_collisions = 5

        
    def gauss(self, x, a, b, c):
        return a * np.exp(-((x+b)**2)/(2*c**2))
    
    def gamma_E(self):
        E = 662
        return E

    def init_gamma_pos(self):
        """
        The y position of gamma photons are distributed as a cos(y)^2
        This random number distribution is generated using the rejection method
        The function returns an array of x-y positions
        """
        N = self.N * 3 # x3 for redundancy - for large N, this avoids having to use the while loop
        w = self.sourceA
        const = 1.
        n = np.pi/w
        
        y_i = np.random.uniform(-w/2, w/2, N)
        P_y = np.cos(n*y_i)**2
        p_i = np.random.uniform(0, const, N)
        
        mask = P_y > p_i # Accepts y_i given this condition
        y_pos = y_i[mask]
        y_pos = y_pos[0:self.N]
        
        # Use a while loop to fill out the y_pos array if its length is less than self.N
        while len(y_pos) < self.N:
            
            y_i = np.random.uniform(-w/2, w/2)
            P_y = np.cos(n*y_i)**2
            p_i = np.random.uniform(0, const)
            
            mask = P_y > p_i # Accepts y_i given this condition
            if mask:
                # Convert to list to allow appending
                y_pos = y_pos.tolist()
                y_pos.append(y_i)
                y_pos = np.array(y_pos)
                
        # Try uniform distribution:
        # y_pos = np.random.uniform(-w/2, w/2, self.N)
        
        # plt.hist(y_pos, bins = 50)
        # Equation of the circle centered at the origin - x^2 + y^2 = r^2
        y = y_pos
        r = self.targetD/2
        x_pos = -np.sqrt(r**2 - y**2) # The gamma particles are incident to the left of the y-axis
        
        # x-y positon
        pos = np.array([x_pos.T, y_pos.T]).T
        return pos
            
    def direction_vec(self, parallel = True):
        """
        Returns an array of direction vectors for each photon
        """        
        direction = np.array([[1], [0]], dtype = 'float')
        dir_array = np.ones(self.N)
        dir_array = np.kron(dir_array, direction).T
        
        if not parallel:
            # Randomize the directions slightly
            scale = 0.01
            rands = np.random.uniform(-scale, scale, size = (self.N, 2))
            dir_array = dir_array + rands
        
        dir_array = np.array([v / np.sqrt(np.sum(v**2)) for v in dir_array])
        
        return dir_array
    
    def circle(self, x, pos, dir_vec):
        """
        "if" statements needed because upper and lower half of the circle are 
        defined separately.
        """
        r = self.targetD/2
        
        # Check where the line intersects the x-axis
        lim1 = -r
        lim2 = r
        try:
            r_intersect = bisect(self.linear_photon, lim1, lim2, args = tuple([pos, dir_vec]))
            r_bool = (abs(r_intersect) <= r)
        except:
            r_bool = False
        
        # print(r_bool)
        
        if ((dir_vec[1] >= 0) and (r_bool)) or ((pos[1] >= 0) and (not r_bool)):
            y = np.sqrt(r**2 - x**2) 
        elif ((dir_vec[1] < 0) and (r_bool)) or ((pos[1] < 0) and (not r_bool)): 
            y = -np.sqrt(r**2 - x**2) 
        return y
    
    def linear_photon(self, x, pos, dir_vec):
        y_lin = (dir_vec[1]/dir_vec[0]) * x + pos[1] - (dir_vec[1]/dir_vec[0]) * pos[0]
        return y_lin
    
    def circle_diff(self, x, pos, dir_vec):
        """
        The trajectory of a photon is modelled as a linear function
        This function returns the difference between the equation of a circle
        and a straight line.
        """
        # y_lin = (dir_vec[1]/dir_vec[0]) * x + pos[1] - (dir_vec[1]/dir_vec[0]) * pos[0]   
        y_lin = self.linear_photon(x, pos, dir_vec)
        y = self.circle(x, pos, dir_vec) - y_lin
        return y
    
    def calc_x_intercept(self, pos, dir_vec):
        """
        Calculate the maximum length the photon can travel in target without 
        interacting
        
        In order to ensure bisect works all the time, the x-limit is extended beyond 
        the radius of the circle by a small amount. This causes 
        """
        if dir_vec[0] >= 0:
            lim1 = pos[0]
            lim2 = self.targetD/2 + 1e-10 # This needs to be smaller than 1e-9
        elif dir_vec[0] < 0:
            lim1 = pos[0]
            lim2 = -self.targetD/2 - 1e-10
            
        x_int = bisect(self.circle_diff, lim1, lim2, args = tuple([pos, dir_vec]))
        y_int = self.linear_photon(x_int, pos, dir_vec)
        
        # Account for the bisect function solution when the particle is on the circle's edge
        if (x_int == pos[0]) or abs(x_int - pos[0]) < 1e-9:
            # Move the photon a small amount into the circle along the direction specified
            small_val = 1e-9 # The scale of the system is O(0.01), so this doesn't cause problems
            pos += dir_vec * small_val
            x_int, y_int = self.calc_x_intercept(pos, dir_vec)
        
        return np.array([x_int, y_int])
        
    def max_path_in_target(self, pos, dir_vec):
        """
        Computes the maximum path a photon can travel in the target for a given 
        direction and position
        """
        init_pos = pos
        final_pos = self.calc_x_intercept(init_pos, dir_vec)
        max_len = np.sqrt((init_pos[0] - final_pos[0])**2 + (init_pos[1] - final_pos[1])**2)
        return max_len
        
    def is_in_target(self, pos):
        """
        Checks if the photon is in the target 
        """
        radius = np.sqrt(pos[0]**2 + pos[1]**2)
        boolean = radius <= self.targetD/2
        return boolean
            
    def sigma_compton(self, Z, E):
        """
        E should be in keV
        Z: Atomic number of target
        r_e is in meters
        """
        r_e = 2.8179e-15 # meters
        m_e = 9.11 * 10**-31
        c = 3e8
        e_rest_mass_E = m_e*c**2
        k = 1.6e-19*1000*E/e_rest_mass_E
        
        coeff = (Z*2*np.pi*r_e**2)
        term1 = ((1+k)/k**2) * (2*(1+k)/(1+2*k) - np.log(1+2*k)/k)
        term2 = np.log(1+2*k)/(2*k) - (1+3*k)/(1+2*k)**2
        sig_C = coeff * (term1 + term2)
        #print(e_rest_mass_E)
        return sig_C
    
    def mean_free_path(self, sig, n):
        """
        Parameters
        sig: Cross section
        length: The length a photon can travel through the target given it doesn't interact
        n: Number density of target
        """
        return 1/(sig * n)
    
    def N0(self, mfp, length):
        """
        Parameters
        sig: Cross section
        length: The length a photon can travel through the target given it doesn't interact
        n: Number density of target
        """
        return length / mfp        
    
    def poisson(self, mean, event):
        """
        Poisson distribution for a system with a mean mean. 
        event is the number of events you want to compute the probabiltiy for
        """
        poiss = ((mean**event) / mt.factorial(event)) * np.exp(-mean)
        return poiss
        
    def plot_photon(self):
        poss = self.init_gamma_pos()
        pos = poss[20]
        poss2 = poss.T
        
        dir_vec_arr = self.direction_vec()
        dir_vec = dir_vec_arr[20]
        
        x_int, y_int = self.calc_x_intercept(pos, dir_vec)
        
        x_arr = np.linspace(pos[0], x_int, 10000)
        y_arr = self.linear_photon(x_arr, pos, dir_vec)
        y_circ = self.circle(x_arr, pos, dir_vec)
    
        # Plot this
        plt.plot(poss2[0], poss2[1], '.')
        plt.plot(x_arr, y_arr, '-')
        plt.plot(x_arr, y_circ)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axhline(linestyle = '--', color = 'black', alpha = 0.4)
        plt.xlim(-0.011, 0.011)
        plt.ylim(-0.011, 0.011)
        plt.xlabel("x direction (m)")
        plt.ylabel("y direction (m)")
        # plt.show()
        
class run_sim(sig_sim):
    def __init__(self):
        sig_sim.__init__(self)
        
    def compton_curve(self, x, E):
        b = 662000*1.6e-19 / (9.11e-31*(3e8)**2)
        a = E
        return a / (1 + b * (1 - np.cos(x)))
    
    def diff_cross_sec(self, theta, E):
        """
        Differential cross section
        """
        r_e = 2.8179e-15 # meters
        m_e = 9.11 * 10**-31
        c = 3e8
        e_rest_mass_E = m_e*c**2
        alph = 1.6e-19*1000*E/e_rest_mass_E
        
        term1 = (r_e**2/2) * ((1+np.cos(theta)**2)/(1+alph*(1-np.cos(theta))**2))
        term2 = 1 + (alph*(1-np.cos(theta)))**2 / ((1+np.cos(theta)**2) * (1 + alph * (1 - np.cos(theta))))
        return term1 * term2        

    def d_sig_d_theta(self, theta, E):
        """
        Cross section derivative in terms of scattering angle
        """
        diff_sigma = self.diff_cross_sec(theta, E)
        diff_sig_theta = 2 * np.pi * np.sin(theta) * diff_sigma
        return diff_sig_theta
    
    def angular_PDF(self, theta, E):
        """
        This works
        """
        Z = self.Z
        sigma = self.sigma_compton(Z, E)
        diff_sig_theta = self.d_sig_d_theta(theta, E) 
        return (1/sigma) * diff_sig_theta
    
    def delta_angle(self, E):
        """
        Computes the angle due to the change in theta 
        """
        E = 662
        theta = np.linspace(0, np.pi, 300)
        P = obj.angular_PDF(theta, E)
        const = max(P) # Choosing a ceiling for the rejection method
        
        mask = False
        while mask == False:
            # Rejection method for choosing the angle
            y_i = np.random.uniform(0, np.pi)
            P_y = obj.angular_PDF(y_i, E)
            p_i = np.random.uniform(0, const)
            
            mask = P_y > p_i # Accepts y_i given this condition
            theta_accept = y_i
        
        # Choose a sign for the angle
        sign = np.sign(np.random.uniform(-1, 1))
                
        return sign * theta_accept
    
    def change_dir(self, theta, dir_vec):
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        new_dir = rot_matrix @ dir_vec
        new_dir = new_dir / np.sqrt(np.sum(new_dir**2))
        
        if new_dir[0] == 1.0:
            new_dir[0] += 1e-9
            new_dir = new_dir / np.sqrt(np.sum(new_dir**2))
        elif new_dir[1] == 1.0:
            new_dir[0] += 1e-9
            new_dir = new_dir / np.sqrt(np.sum(new_dir**2))
            
        return new_dir
    
    def change_energy(self, theta, E):
        """
        Changes energy due to the compton effect equation
        """
        return self.compton_curve(theta, E)
    
    def log_func(self, x, mfp):
        return -mfp * np.log((1-x)/mfp)
    
    def output_path_travelled(self, mfp, max_len):
        """
        This this distribution can be generated using the transformation method
        The transformation method is generally more efficient than the rejection method
        
        Equation for the distributions of paths generated.
        """
        # For the mfp values of our interest, the x_intercept is between -10, 10
        # lim1 = -10
        # lim2 = 10
        # x_left = bisect(self.log_func, lim1, lim2, args = tuple([mfp]))
        
        # # Impose upper bound on y_i
        # y_i = max_len + 1
        # while y_i > max_len:
        #     x = np.random.uniform(x_left, 1)
        #     y_i = self.log_func(x, mfp)
        
        # Try rejection method instead
        const = 1
        mask = False
        while not mask:
            y_i = np.random.uniform(0, max_len)
            P_y = np.exp(-y_i/mfp)**2
            p_i = np.random.uniform(0, const)
            mask = P_y > p_i # Accepts y_i given this condition
        
        y_pos = y_i
            
        return y_pos
    
    def change_position(self, pos, dir_vec, move_dist):
        new_pos = pos + dir_vec * move_dist
        return new_pos
         
    def final_N0(self, E, Z, length):
        """
        Combine previous functions to compute stuff for collisions 
        """
        sigma = self.sigma_compton(Z, E)
        mfp = self.mean_free_path(sigma, self.n_Al)
        N_cols = self.N0(mfp, length)        
        return N_cols
    
    def to_collide_or_not_to(self, p):
        rand_num = np.random.uniform()
        if rand_num < p:
            collide = True
        else:
            collide = False
        return collide
    
    def big_circle(self, dist, radius, pos, dir_vec):
        """
        Used to propagate the photons to end point
        """
        new_pos = pos + dist * dir_vec
        mag_new_pos = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
        # Radii difference 
        r_diff = radius - mag_new_pos
        return r_diff
    
    def run_for_plots(self):
        pos_arr = self.init_gamma_pos()
        directions = self.direction_vec()
        Z = self.Z
        
        # Setup the simulation
        max_cols = self.maximum_collisions 
        in_target = True # The loop continues while the photon is in aluminium
        
        positions = []
        E_arr = []
        for i in range(self.N):
            
            # if (i % 100) == 0:
            print(i)
            
            pos = pos_arr[i]
            dir_vec = directions[i]
            
            # Save the positions 
            positions.append([])
            positions[i].append(pos)
            
            # Collision counter
            current_cols = 0
            length = self.max_path_in_target(pos, dir_vec)
            in_target = self.is_in_target(pos)
            
            # print("NEW IF LOOP")
            E = 662
            while in_target and (max_cols >= current_cols):
                N_cols = self.final_N0(E, Z, length)
                p_1col = 1 - self.poisson(N_cols, 0)
                col_bool = self.to_collide_or_not_to(p_1col)
                
                # print("N_cols:", N_cols)
                # print("Length:", length)
                # print("P:", p_1col)
                # print("E", E)
                
                if col_bool:
                    # Update position
                    sig = self.sigma_compton(self.Z, E)
                    mfp = self.mean_free_path(sig, self.n_Al)
                    move_dist = self.output_path_travelled(mfp, length)
                    pos = self.change_position(pos, dir_vec, move_dist)
                    
                    # count_all_cols += 1
                    # print(count_all_cols)
                    
                    # positions[i].append(pos) 
                    
                    # Update the other parameters
                    angular_deflection = self.delta_angle(E)
                    # Update direction
                    dir_vec = self.change_dir(angular_deflection, dir_vec)
                    # Update energy
                    E = self.change_energy(angular_deflection, E)
                    # Update collision counter
                    current_cols += 1
                    # Update "length to the circle"
                    length = self.max_path_in_target(pos, dir_vec) 
                    
                if (current_cols == max_cols) or not col_bool:
                    # Propagate the photon out of the target 
                    radius = self.T_to_D_D
                    small_amount = self.targetD
                    lim1 = 0
                    lim2 = radius + small_amount
                    
                    dist = bisect(self.big_circle, lim1, lim2, args = tuple([radius, pos, dir_vec]))
                    
                    pos = self.change_position(pos, dir_vec, dist)
                    
                    positions[i].append(pos)
                else:
                    # Save position
                    positions[i].append(pos)
                
                # Check if the photon is still in the aluminium target
                in_target = self.is_in_target(pos)
                
            # Append to energy array
            E_arr.append(E)
        
        # for i in positions:
        #     if len(i) == 4:
        #         # print(i)
        #         i = np.array(i)
        #         i = i.T
        #         plt.plot(i[0], i[1])
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.axhline(linestyle = '--', color = 'black', alpha = 0.4)
        # plt.xlim(-0.011, 0.011)
        # plt.ylim(-0.011, 0.011)
        # plt.show()
            
        # quant = [len(positions[i]) for i in range(self.N)]
        # print(quant)
        
        return np.array(positions), np.array(E_arr)
    
    def pos_to_angle(self, pos_array):
        all_angles = []
        for i in range(len(pos_array)):
            end_pos = pos_array[i][-1]
            # np.arctan(y, x) equivalent to arctan(y/x)
            angle = np.arctan2(end_pos[1], end_pos[0])
            all_angles.append(angle)
        
        all_angles = np.array(all_angles)
        return all_angles
            
class detectors(run_sim):
    """
    Detectors can be modelled as a line of the detector aperture width. 
    The detector at 0 angular width will have to be tilted by a small amount to
    allow bisection to work. 
    """
    
    def gamma_detector(self):
        # l = r*theta
        detector_ang_res = self.detectorA / self.T_to_D_D 
        print(detector_ang_res)
        angle_pos_bins = np.arange(np.pi/2 + detector_ang_res, 0, -detector_ang_res)[::-1]
        # angle_neg_bins = np.arange(-np.pi/2 - detector_ang_res, 0, detector_ang_res)
        
        return angle_pos_bins
    
    
class visualization(run_sim):
    def scattering_angle_dist(self):
        """
        Plots the scattering angle distribution derived from the differential 
        cross section. 
        """
        all_ang = []
        for i in range(10000):
            all_ang.append(self.delta_angle(662))
        
        plt.hist(all_ang, 200)
        plt.xlabel("Angle (rads)")
        plt.ylabel("Frequency")
        plt.show()
    
    def mfp_vs_energy(self):
        E_arr = np.linspace(1, 1000, 1000)
        all_paths = []
        n = self.n_Al
        for i in E_arr:
            sig = self.sigma_compton(obj.Z, i)
            mfp = self.mean_free_path(sig, n)
            all_paths.append(mfp)
        
        plt.plot(E_arr, all_paths)
        plt.xlabel("Energy (keV)")
        plt.ylabel("Mean Free Path (m)")
        plt.minorticks_on()
        plt.grid(which = 'minor', alpha = 0.2)
        plt.grid(which = 'major')
        # plt.savefig("mfp_vs_energy.png", dpi = 300, bbox_inches="tight")
        plt.show()
        
        
    def plot_trajectories(self, pos_array):        
        fig, ax = plt.subplots()
        circle1 = plt.Circle((0, 0), self.targetD/2, color='grey', alpha = 0.3)
        ax.add_patch(circle1)
        red_ones = 0
        for i in pos_array:
            i = np.array(i)
            i = i.T
            x_arr = i[0]
            y_arr = i[1]
            
            if len(x_arr) >= 4:
                line_colour = 'red'
                red_ones += 1
            if len(x_arr) == 3:
                red_ones += 1
                line_colour = 'blue'
            elif len(x_arr) < 3:
                line_colour = 'black'
            
            # if (line_colour == 'red'): #or (line_colour == 'blue'):
            ax.plot(x_arr, y_arr, color = line_colour)
            plt.xlabel("x position (m)")
            plt.ylabel("y position (m)")
        plt.gca().set_aspect('equal', adjustable='box')
        ax.axhline(linestyle = '--', color = 'black', alpha = 0.4)
        plt.xlim(-0.011, 0.011)
        plt.ylim(-0.011, 0.011)
        # outer = self.T_to_D_D + 0.01
        # plt.xlim(-outer, outer)
        # plt.ylim(-outer, outer)
        plt.show()
        print("Total collided photons:", red_ones)
        
    def plot_detector_positions(self):
        """
        Plotting the detector position with the rest of the setup
        """
        fig, ax = plt.subplots()
        circle1 = plt.Circle((0, 0), self.targetD/2, color='grey', alpha = 0.3)
        ax.add_patch(circle1)
        plt.gca().set_aspect('equal', adjustable='box')
        outer = self.T_to_D_D + 0.01
        plt.xlim(-outer, outer)
        plt.ylim(-outer, outer)
        plt.show()
        
    def plot_post_scatter_angles(self, pos_array):
        # pos_array = self.run_for_plots()
        all_angles = []
        for i in range(len(pos_array)):
            end_pos = pos_array[i][-1]
            # np.arctan(y, x) equivalent to arctan(y/x)
            angle = np.arctan2(end_pos[1], end_pos[0])
            all_angles.append(angle)
        
        all_angles = np.array(all_angles)
        mask = (all_angles > 5 * np.pi/180) | (all_angles < -5 * np.pi/180)
        all_angles = all_angles[mask]
          
        # Plot histogram
        plt.hist(all_angles, bins = 100)
        plt.xlabel("Angle (rads)")
        plt.ylabel("Frequency")
        plt.minorticks_on()
        plt.grid(which = 'minor', alpha = 0.2)
        plt.grid(which = 'major')
        # plt.savefig("freq_vs_angle_simulated.png", dpi = 300, bbox_inches="tight")
        plt.show()
        plt.show()
            
        
        
if __name__ == '__main__':
    obj = detectors()
    
    E = 662
    Z = obj.Z
    
    pos_array, E_arr = obj.run_for_plots()
    angles = obj.pos_to_angle(pos_array)
    data = np.array([angles, E_arr]).T
    
    dF = pd.DataFrame(data, columns = ['angles', 'energy'])
    dF.to_csv("max_col5_5M.csv")
    
    # plt.plot(angles, E_arr, '.', alpha = 0.6)
    
    # x = np.linspace(0, np.pi, 200)
    # comp_E = obj.compton_curve(x, E)
    # plt.plot(x, comp_E, '-', lw = 2)
    
    # obj.plot_trajectories(pos_array)
    # obj.plot_post_scatter_angles(pos_array)
    
    
