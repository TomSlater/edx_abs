# -*- coding: utf-8 -*-
"""
Created on Wed Sep 02 15:43:56 2015

@author: Tom
"""

import numpy as np
from skimage.transform import iradon, iradon_sart
#import em_io
import astra
from skimage.feature import register_translation
from scipy.ndimage import interpolation, filters
import pylab as plt
import time

def reconstruct(tilt_data,tilt_angles,algorithm='ASTRA_SIRT',iterations=20,geometry='parallel3d'):
    '''Function to reconstruct a tilt series.
    
    Data should be in the format (Angles,X,Y), where the tilt axis is situated about the mid column of the data.'''
    
    t0 = time.time()
    data_shape = np.shape(tilt_data)
    y_size = data_shape[1]
    
    recon = np.zeros((data_shape[2],y_size,data_shape[2]))
    #sino = np.zeros(data_shape)
    
    #Reconstruction using Filtered/Weighted Backprojection from skimage (check how filtering is done)
    if algorithm == 'SKI_FBP' or 'SKI_WBP':
        for y in range(0,y_size-1):
            recon[:,y,:] = iradon(np.rot90(tilt_data[:,y,:]), theta = tilt_angles,output_size = data_shape[2])
            
            #This is supposed to reorder axis to orientation for projection but not sure the x and y are correct
            for y in range(0,y_size-1):
                recon[:,y,:] = np.rot90(recon[:,y,:])
                recon[:,y,:] = np.rot90(recon[:,y,:])
                #recon[:,y,:] = np.rot90(recon[:,y,:])
            
    if algorithm == 'SKI_SART':
        for y in range(0,y_size-1):
            recon[:,y,:] = iradon_sart(np.rot90(tilt_data[:,y,:]), theta = tilt_angles,clip=(0,np.max(tilt_data)))
            
        if iterations > 1:
            for it in range(iterations-1):
                print("Iteration number "+str(it+2)+" in progress.")
                for y in range(0,y_size-1):
                    recon[:,y,:] = iradon_sart(np.rot90(tilt_data[:,y,:]), theta = tilt_angles,image=recon[:,y,:],clip=(0,np.max(tilt_data)))
                    
        #This is supposed to reorder axis to orientation for projection but not sure the x and y are correct
        for y in range(0,y_size-1):
            recon[:,y,:] = np.rot90(recon[:,y,:])
            recon[:,y,:] = np.rot90(recon[:,y,:])
            #recon[:,y,:] = np.rot90(recon[:,y,:])
                    
    if algorithm == 'ASTRA_SIRT':
        recon = astrarecon(tilt_data,tilt_angles,iterations,geometry)
      
        '''for z in xrange(0,data_shape[2]):
            recon[:,:,z] = np.rot90(recon[:,:,z])
            recon[:,:,z] = np.rot90(recon[:,:,z])
            recon[:,:,z] = np.rot90(recon[:,:,z])'''
        
    print("Reconstruction completed in {} seconds.".format(time.time() - t0))
            
    return(recon)
    
def astrarecon(tilt_data,tilt_angles,iterations=1,geometry='parallel3d',SO_dist=1.0,OD_dist=1.0):
    proj_shape = np.shape(tilt_data)
    recon_shape = (proj_shape[2],proj_shape[2],proj_shape[1])
    
    vol_geom = astra.create_vol_geom(recon_shape)

    angles = np.pi*tilt_angles/180
    
    if geometry == 'parallel3d':
        proj_geom = astra.create_proj_geom(geometry, 1.0, 1.0, proj_shape[1], proj_shape[2], angles)
        cfg = astra.astra_dict('SIRT3D_CUDA')
    elif geometry == 'cone':
        proj_geom = astra.create_proj_geom(geometry, 1.0, 1.0, proj_shape[1], proj_shape[2], angles, SO_dist, OD_dist)
        cfg = astra.astra_dict('FDK_CUDA')
        
    proj_id = astra.data3d.create('-proj3d', proj_geom, np.swapaxes(tilt_data,0,1))
    
    rec_id = astra.data3d.create('-vol', vol_geom)
    
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    
    alg_id = astra.algorithm.create(cfg)
    
    astra.algorithm.run(alg_id, iterations)
    
    rec = astra.data3d.get(rec_id)
    
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)
    
    return(rec)
    
def project(reconstruction,angles):
    '''Function to project a reconstruction'''
    proj_list = []
    
    for angle in angles:
        rot_recon = rotate3d(reconstruction,90,0,0,False)
        rot_recon = rotate3d(rot_recon,0,angle,0,False)
        rot_recon = rotate3d(rot_recon,-90,0,0,False)
        projection = np.zeros((np.shape(rot_recon)[0],np.shape(rot_recon)[1]))
        
        for x in range(np.shape(rot_recon)[0]):
            for y in range(np.shape(rot_recon)[1]):
                projection[x,y] = np.sum(rot_recon[x,y,:])
                #projection = np.rot90(projection)
                
        proj_list.append(projection)
            
    return(proj_list)
    
def series_align(im_series,align_output=[],start='Mid',smooth=True,smooth_window='3',sobel=True):
    '''Function to align a series of images.'''
    if align_output == []:
        series_dim = len(im_series)
        
        filtered_series = []
        
        for i in range(series_dim):
            filtered_series.append(im_series[i].copy())
        
        align_output = []
        
        if smooth == True:
            for i in range(series_dim):
                filtered_series[i] = filters.gaussian_filter(filtered_series[i],3)
        
        if sobel == True:
            for i in range(series_dim):
                filtered_series[i] = filters.sobel(filtered_series[i])
        
        #Align from first image
        if start == 'First':
            align_output.append(register_translation(filtered_series[0], filtered_series[0],100))
            for i in range(series_dim-1):
                align_output.append(register_translation(filtered_series[i], filtered_series[i+1],100))
                align_output[i+1][0][0] = align_output[i+1][0][0] + align_output[i][0][0]
                align_output[i+1][0][1] = align_output[i+1][0][1] + align_output[i][0][1]
        
        #Align from mid-image
        if start == 'Mid':
            
            #Ensure compatibility with Pyton 2 and 3
            if series_dim % 2 == 0:
                mid_point = series_dim / 2
            else:
                mid_point = series_dim // 2
            
            align_output.append(register_translation(filtered_series[mid_point], filtered_series[mid_point], 100))
            
            for i in range(mid_point,0,-1):
                align_output.append(register_translation(filtered_series[i], filtered_series[i-1], 100))
                align_output[mid_point-i+1][0][0] = align_output[mid_point-i+1][0][0] + align_output[mid_point-i][0][0]
                align_output[mid_point-i+1][0][1] = align_output[mid_point-i+1][0][1] + align_output[mid_point-i][0][1]
                
            align_output = list(reversed(align_output))
            
            for i in range(mid_point,series_dim-1):
                align_output.append(register_translation(filtered_series[i], filtered_series[i+1], 100))
                align_output[i+1][0][0] = align_output[i+1][0][0] + align_output[i][0][0]
                align_output[i+1][0][1] = align_output[i+1][0][1] + align_output[i][0][1]
        
    #Apply calculated shifts to the image series
    shifted_im_series = []
    im_count = 0
    for im in im_series:
        shifted_im_series.append(interpolation.shift(im,align_output[im_count][0]))
        im_count = im_count + 1
        
    shifted_im_series = np.asarray(shifted_im_series)
        
    return(shifted_im_series, align_output)
    
def tiltaxisalign(im_series,tilt_angles,shift_and_tilt=('hold','hold')):
    series_shape = np.shape(im_series)
    
    new_series = im_series.copy()
    final_series = im_series.copy()
    
    #deg0_int = input('Which image is the 0 degree image? ')
    midy = int(series_shape[1]/2)
    
    axis_shift = shift_and_tilt[0]
    axis_tilt = shift_and_tilt[1]
    
    if axis_shift == 'hold':
        shift_continue = 1
    
        while shift_continue == 1:
            plt.imshow(iradon(np.rot90(new_series[:,midy,:]), theta = tilt_angles,output_size = series_shape[2]))
            plt.show()
            axis_shift = input('By how many pixels from the original mid-point should the tilt axis be shifted? ')
            for i in range(series_shape[0]):
                new_series[i,:,:] = interpolation.shift(im_series.copy()[i,:,:],(0,axis_shift))
                
            shift_continue = input('Would you like to apply further image shifts (1 for yes, 0 for no)? ')
                
    for i in range(series_shape[0]):
        final_series[i,:,:] = interpolation.shift(final_series[i,:,:],(0,axis_shift))
        
    topy = int(series_shape[1]/4)
    bottomy = int(3*series_shape[1]/4)

    if axis_tilt == 'hold':
        tilt_series = new_series
        tilt_continue = 1
        while tilt_continue == 1:
            plt.imshow(iradon(np.rot90(new_series[:,topy,:]), theta = tilt_angles,output_size = series_shape[2]))
            plt.show()
            plt.imshow(iradon(np.rot90(new_series[:,bottomy,:]), theta = tilt_angles,output_size = series_shape[2]))
            plt.show()
            
            axis_tilt = input('By what angle from the original y axis (in degrees) should the tilt axis be rotated? ')
            
            if axis_tilt != 0:
                for i in range(series_shape[0]):
                    new_series[i,:,:] = interpolation.rotate(tilt_series.copy()[i,:,:],axis_tilt,reshape=False)
                    
            tilt_continue = input('Would you like to try another tilt angle (1 for yes, 0 for no)? ')
                    
    if axis_tilt != 0:
        for i in range(series_shape[0]):
            final_series[i,:,:] = interpolation.rotate(final_series[i,:,:],axis_tilt,reshape=False)
            
    shift_and_tilt = (axis_shift,axis_tilt)
            
    return(final_series, shift_and_tilt)
    
def bin3d(data_in,x_bin=2,y_bin=2,z_bin=2,binning='Average'):
    '''Function to bin 3D data by arbitrary integers in each dimensions'''
    
    dshape = np.shape(data_in)
    
    #Check all dimensions are divisible by the binning required
    if dshape[0] % x_bin != 0 or dshape[1] % y_bin != 0 or dshape[2] % z_bin != 0:
        raise ValueError("One or more axes of the input data is not divisible by the binning integers!")
    
    bin_sum = 0
    data_out = np.zeros((dshape[0]/x_bin,dshape[1]/y_bin,dshape[2]/z_bin))
    for x in range(dshape[0]/x_bin):
        for y in range(dshape[1]/y_bin):
            for z in range(dshape[2]/z_bin):
                
                #Check if binnning should average or sum voxel values
                if binning=='Average':
                    data_out[x,y,z] = bin_sum / (x_bin*y_bin*z_bin)
                if binning=='Sum':
                    data_out[x,y,z] = bin_sum
                bin_sum = 0
                
                for x_i in range(x_bin):
                    for y_i in range(y_bin):
                        for z_i in range(z_bin):
                            bin_sum = bin_sum + data_in[x*x_bin+x_i,y*y_bin+y_i,z*z_bin+z_i]
    
    return(data_out)
    
def rotate3d(data_in,alpha=0,beta=0,gamma=0,reshape=True,cval_in=0.0):
    '''Function to rotate a numpy array by the Euler angles alpha, beta, gamma.'''
    
    dshape = np.shape(data_in)
    
    rot_data_shape = np.shape(interpolation.rotate(data_in[:,:,0],gamma,reshape=reshape,order=1))
    rot_data_gamma = np.zeros((rot_data_shape[0],rot_data_shape[1],dshape[2]))
    
    for z in range(dshape[2]):
        rot_data_gamma[:,:,z] = interpolation.rotate(data_in[:,:,z],gamma,reshape=reshape,cval=cval_in,order=1)
    
    rot_data_shape = np.shape(interpolation.rotate(rot_data_gamma[:,0,:],beta,reshape=reshape,order=1))
    rot_data_beta = np.zeros((rot_data_shape[0],np.shape(rot_data_gamma)[1],rot_data_shape[1]))
        
    for y in range(dshape[1]):
        rot_data_beta[:,y,:] = interpolation.rotate(rot_data_gamma[:,y,:],beta,reshape=reshape,cval=cval_in,order=1)
        
    rot_data_shape = np.shape(interpolation.rotate(rot_data_beta[:,:,0],alpha,reshape=reshape,order=1))
    rot_data_alpha = np.zeros((rot_data_shape[0],rot_data_shape[1],np.shape(rot_data_beta)[2]))
    
    for z in range(dshape[2]):
        rot_data_alpha[:,:,z] = interpolation.rotate(rot_data_beta[:,:,z],alpha,reshape=reshape,cval=cval_in,order=1)
        
    return(rot_data_alpha)
    
def generate_sinogram(tilt_series):
    sino = []
    
    for n in range(len(tilt_series)):
        mid = np.shape(tilt_series[n])[0]/2
        sino.append(tilt_series[n][100,:])
        
    return(sino)
    
def ss_align(series,short_series,ss_shifts=[]):
    
    alignments_init = []
    for i in range(len(short_series)):
        j = i*((len(series)-1)//(len(short_series)-1))
        #print(i,j)
        alignments_init.append(register_translation(short_series[i],series[j],100)[0])
    
    if ss_shifts != []:
        alignments_init=ss_shifts
        
    alignments = []
    for i in range(len(series)):
        if i==0 or i%((len(series)-1)/(len(short_series)-1))==0:
            alignments.append(alignments_init[i//((len(series)-1)//(len(short_series)-1))])
        
        else:
            diff = alignments_init[i//((len(series)-1)//(len(short_series)-1))+1]-alignments_init[i//((len(series)-1)//(len(short_series)-1))]
            alignment = alignments_init[i//((len(series)-1)//(len(short_series)-1))]+((i%((len(series)-1)//(len(short_series)-1)))*(diff/float(((len(series)-1)//(len(short_series)-1)))))
            alignments.append(alignment)
    
    return alignments
    
def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

    #def cone_centre_shift(projections, angles, check_range):

def centreshift(data,angles,tiltrange,increment):
    series_shape = np.shape(data)
    new_series = data.copy()
    midy = int(series_shape[1]/2)
    for i in range(series_shape[0]):
        new_series[i,:,:] = interpolation.shift(data.copy()[i,:,:],(0,axis_shift))
        
    
    iradon(np.rot90(data[:,midy,:]), theta = angles,output_size = series_shape[2])
        
    
#if __name__ == "__main__":
    #x = hspy.load('C:/Users/Tom/Documents/TEM data/20150521_SS316needle/EDX Tomo/Cr/Signed_zero/cr_sub_hdr0_2.ali')
    '''x = h5py.File('C:/Users/Tom/Documents/TEM data/20150521_SS316needle/EDX Tomo/Cr/Signed_zero/cr_sub_hdr0_2_bin8.h5', "r")
    tilt_angles = np.linspace(-90.,90.,37)
    x_data = x['/0']
    reconstruction = reconstruct(x_data,tilt_angles)'''
    '''
    x = em_io.load_txt('C:/Users/Tom/Documents/TEM data/20150521_SS316needle/EDX Tomo/Cr/1_-90deg.txt',series=True)
    
    x_data = np.asarray(x)
    
    x_binned = bin3d(x_data,x_bin=1,y_bin=8,z_bin=8,binning='Sum')
    
    ls = list(x_binned)
    
    shifted_ims, shifts = series_align(x_binned,start='Mid')
    
    shifted_ims_array = np.asarray(shifted_ims)
    
    tilt_angles = np.linspace(-90.,90.,37)
    
    ali_recon = reconstruct(shifted_ims_array,tilt_angles)
    unali_recon = reconstruct(x_binned,tilt_angles)
    
    #em_io.savehdf(shifted_ims,'C:/Users/Tom/Documents/TEM data/20150521_SS316needle/EDX Tomo/Cr/Cr_shifted.h5')'''