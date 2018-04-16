# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:10:40 2018

@author: Thomas Slater
"""

import hyperspy.api as hs
import glob
import numpy as np
import tools_3d as t3d

def load_edx_series(directory,elements):
    dic = dict()
    
    for el in elements:
        dic[el] = dict()
        dic['HAADF'] = dict()
        
    for file in glob.glob(directory+'*.emd'):
        s = hs.load(file, signal_type="EDS_TEM")
        
        for i in s:
            if i.metadata.General.title == 'HAADF': 
                haadf = i
            if i.metadata.General.title == 'EDS':
                spec_im = i
        
        spec_im.set_elements(elements)
        spec_im.add_lines()
        
        lines = spec_im.metadata.Sample.xray_lines
        
        dic['HAADF'][haadf.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha] = haadf.data
        
        for line in lines:
            el = line.rpartition('_')[0]
            dic[el][spec_im.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha] = spec_im.get_lines_intensity([line])[0].data
            
    dic2 = dict()
    
    for el in elements:
        el_ls = []
        for key in sorted(dic[el],key=float):
            el_ls.append(dic[el][key])
            
        dic2[el]=np.asarray(el_ls)
        
    haadf_ls = []
    angles = []
    for key in sorted(dic['HAADF'],key=float):
        haadf_ls.append(dic['HAADF'][key])
        angles.append(float(key))
        
    dic2['HAADF'] = np.asarray(haadf_ls)
        
    return(dic2,angles)

def align_edx_series(data_dict,angles):
    aligned = dict()
    
    aligned['HAADF'], alignments = t3d.series_align(data_dict['HAADF'])
    
    for el in data_dict:
        if el != 'HAADF':
            aligned[el], alignments = t3d.series_align(data_dict[el],align_output=alignments)
            
    tilt_aligned = dict()
    tilt_aligned['HAADF'], shift_and_tilt = t3d.tiltaxisalign(aligned['HAADF'],angles)
    
    for el in data_dict:
        if el != 'HAADF':
            tilt_aligned[el], shift_and_tilt = t3d.tiltaxisalign(aligned[el],angles,shift_and_tilt=shift_and_tilt)
            
    return(tilt_aligned)

def recon_edx_series(data_dict,angles):
    recons = {}
    
    for el in data_dict:
        recons[el] = t3d.reconstruct(data_dict[el],angles)
        
    return(recons)