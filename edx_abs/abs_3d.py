# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:10:03 2016

@author: Thomas Slater
"""
import numpy as np

class detectors():
    """Class for storing EDX detector information.
    """
    
    def __init__(self,det_num,angles):
        self.det_num = det_num
        self.detectors = []
        
        for i in range(det_num):
            self.detectors.append({'angles':{'elevation':angles[i]['elevation'],'azimuth':angles[i]['azimuth']}})
            
    def add_detector(self,det_num,angles):
        pre_det_num = self.det_num
        self.det_num = pre_det_num+det_num
        
        for i in range(pre_det_num,det_num):
            self.detectors.append({'angles':{'elevation':angles[i]['elevation'],'azimuth':angles[i]['azimuth']}})

def composition_map(element_maps,k_factors,elements):
    """Function to calculate maps of composition from intensity maps and k_factors.
    
    element_maps should be a list of maps of elements 1,2,3 etc..
    k_factors should be an array where the k factor is contained in k_row_column"""
    
    ones = np.ones((np.shape(element_maps[elements[0]])))
    
    number_maps = len(element_maps)
    comp_maps = {}

    for element1 in element_maps:
        denom_sum = np.zeros(np.shape(element_maps[element1]))
        for element2 in element_maps:
            if element2 != element1:
                cx_ci = k_factors[element2+'/'+element1]*(element_maps[element2]/element_maps[element1])
                denom_sum = denom_sum + cx_ci
                
        comp_maps[element1] = ones/(ones+denom_sum)
        #comp_maps[element1][comp_maps==1.0] = 0.0

    return(comp_maps)
    
def k_factor_matrix(element_maps,compositions):
    """Function to calculate a matrix of k factors from single element intensity maps
    and pre-defined composition ratios."""
    
    k_factor_array = np.zeros((len(element_maps),len(element_maps)))
    
    for i in range(len(element_maps)):
        for j in range(len(element_maps)):
            if i==j:
                k_factor_array[i,j] = 1
            else:
                k_factor_array[i,j] = (compositions[i]/float(compositions[j]))*(np.sum(element_maps[j])/np.float(np.sum(element_maps[i])))
    
    return(k_factor_array)
    
def k_factor_matrix_dict(element_maps,compositions):
    """Function to calculate a matrix of k factors from single element intensity maps
    and pre-defined composition ratios."""
    
    k_factor_dict = {}

    for element1 in element_maps:
        for element2 in element_maps:
            if element1==element2:
                k_factor_dict[element1+'/'+element2] = 1
            else:
                k_factor_dict[element1+'/'+element2] = (compositions[element1]/float(compositions[element2]))*(np.sum(element_maps[element2])/np.float(np.sum(element_maps[element1])))
                
    return(k_factor_dict)
    
def density_calc(haadf_recon,density_vals):
    density = np.zeros(np.shape(haadf_recon))
    density = density_vals[0]*(haadf_recon**0.5) - density_vals[1]
    density[haadf_recon<density_vals[2]] = 0

    return density
    
def mass_atten_calc(comp_maps,elements):
    mass_atten_dic = mass_atten_dict.generate_mass_atten_dict()
    
    #Mass attenuation maps
    mass_atten = {}
    
    for xray_peak in mass_atten_dic:
        mass_atten[xray_peak] = np.zeros(np.shape(comp_maps[elements[0]]))
        for element in comp_maps:
            mass_atten[xray_peak] = mass_atten[xray_peak] + comp_maps[element]*mass_atten_dic[xray_peak][element]

    return(mass_atten)
    