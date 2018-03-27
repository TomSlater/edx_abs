# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy
import tools_3d as t3d
from scipy.ndimage import interpolation
import abs_3d

def gpu_abs_im(density, mass_atten, intensity, pix_size, detector):
    density = density.astype(numpy.float32)
    mass_atten = mass_atten.astype(numpy.float32)
    result = numpy.ones_like(density).astype(numpy.float32)
    
    density_rot = t3d.rotate3d(density,beta=detector['angles']['elevation'],reshape=False).astype(numpy.float32)
    mass_atten_rot = t3d.rotate3d(mass_atten,beta=detector['angles']['elevation'],reshape=False).astype(numpy.float32)
    mass_atten_rot[mass_atten_rot < 0.01] = 0
    density_rot[density_rot < 0.01] = 0

    dens_gpu = cuda.mem_alloc(density_rot.nbytes)
    cuda.memcpy_htod(dens_gpu, density_rot)
    
    massatten_gpu = cuda.mem_alloc(mass_atten_rot.nbytes)
    cuda.memcpy_htod(massatten_gpu, mass_atten_rot)
    
    result_gpu = cuda.mem_alloc(result.nbytes)
    cuda.memcpy_htod(result_gpu, result)
    
    
    mod = SourceModule("""
    
        __global__ void my_func(float *dens, float *mass_atten, float dx, float *result)
        {
         int idx = threadIdx.x + blockIdx.x * blockDim.x;
         int idy = threadIdx.y + blockIdx.y * blockDim.y;
         int idz = threadIdx.z + blockIdx.z * blockDim.z;
         int x_width = blockDim.x * gridDim.x;
         int y_width = blockDim.y * gridDim.y;
         int z_width = blockDim.z * gridDim.z;
         int flat_id = idx + x_width * idy + x_width*y_width*idz;
         
         float expo = 0;
         
         for(int idzs = idz; idzs < z_width; idzs++)
         {
         
          int id = idx + x_width * idy + x_width * y_width *idzs;
          expo += dens[id]*mass_atten[id]*dx;
          result[flat_id] = exp(-expo);
         
         }
         
        }
        
    """)
    
    func = mod.get_function("my_func")
    
    grid_size = (numpy.int(density.shape[0]/4), numpy.int(density.shape[1]/4), numpy.int(density.shape[2]/4))
    block_size = (4, 4, 4)
    
    func(dens_gpu, massatten_gpu, pix_size, result_gpu, block = block_size, grid = grid_size)
    
    result = numpy.empty_like(density)
    
    cuda.memcpy_dtoh(result, result_gpu)
    
    result = t3d.rotate3d(result, beta=-detector['angles']['elevation'],reshape=False).astype(numpy.float32)
    
    result2d = numpy.ones((density.shape[0],density.shape[1]))
    
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            if mass_atten[x,y,:].sum() > 0.01:
                result_col = result[x,y,:]*intensity[x,y,:]
                result2d[x,y] = (numpy.sum(result_col[mass_atten[x,y,:]>0])) / numpy.sum(intensity[x,y,:])

    del result
    
    return(result2d)

def abs_im(density, mass_atten, intensity, pix_size, detector):
    
    density = density.astype(numpy.float32)
    mass_atten = mass_atten.astype(numpy.float32)
    result = numpy.ones_like(density).astype(numpy.float32)
    
    density_rot = t3d.rotate3d(density,beta=detector['angles']['elevation'],reshape=False).astype(numpy.float32)
    mass_atten_rot = t3d.rotate3d(mass_atten,beta=detector['angles']['elevation'],reshape=False).astype(numpy.float32)
    mass_atten_rot[mass_atten_rot < 0.01] = 0
    density_rot[density_rot < 0.01] = 0
    
    for x in density.shape[0]:
        for y in density.shape[1]:
            for z in density.shape[2]: #I need to check the index here
                result[x,y,z] = numpy.exp(density[x,y,z]*mass_atten[x,y,z]*pix_size) + result[x,y,z-1]
    
    result = t3d.rotate3d(result, beta=-detector['angles']['elevation'],reshape=False).astype(numpy.float32)
    
    result2d = numpy.ones((density.shape[0],density.shape[1]))
    
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            if mass_atten[x,y,:].sum() > 0.01:
                result_col = result[x,y,:]*intensity[x,y,:]
                result2d[x,y] = (numpy.sum(result_col[mass_atten[x,y,:]>0])) / numpy.sum(intensity[x,y,:])

    del result
    
    return(result2d)
    
def gpu_abs_2d(density,mass_atten,intensity,pix_size,detectors):
    dens = []
    mass_attens = []
    intens = []
    det = []
    av_det = numpy.zeros_like(density)
    
    for i in range(detectors.det_num):
        dens.append(t3d.rotate3d(density,detectors.detectors[i]['angles']['azimuth'],reshape=False).astype(numpy.float32))
        mass_attens.append(t3d.rotate3d(mass_atten,detectors.detectors[i]['angles']['azimuth'],reshape=False).astype(numpy.float32))
        intens.append(t3d.rotate3d(intensity,detectors.detectors[i]['angles']['azimuth'],reshape=False).astype(numpy.float32))
        det.append(gpu_abs_im(dens[i],mass_attens[i],intens[i],pix_size,detectors.detectors[i]))
        av_det = ((av_det*i)+interpolation.rotate(det[i],-detectors.detectors[i]['angles']['azimuth'],reshape=False,cval=1.0,order=1))/(i+1)

    return(av_det)
    
def gpu_acf_series(density, mass_atten, intensity, angles, pix_size, detectors=None):
    if detectors==None:
        detectors = abs_3d.detectors(det_num=2,angles=[{'elevation':15,'azimuth':135},
                                                       {'elevation':15,'azimuth':225}])
    
    abs_series = []

    for ang in angles:
        dens_ang = t3d.rotate3d(density,alpha=90,beta=-ang,gamma=-90,reshape=False)
        mass_atten_ang = t3d.rotate3d(mass_atten,alpha=90,beta=-ang,gamma=-90,reshape=False)
        intens_ang = t3d.rotate3d(intensity,alpha=90,beta=-ang,gamma=-90,reshape=False)
        
        abs_im_ang = gpu_abs_2d(dens_ang,mass_atten_ang,intens_ang,pix_size,detectors)
        
        abs_series.append(abs_im_ang)
        
    return(abs_series)
    
def full_acf_series(density,mass_atten,intensity,angles,pix_size,detectors=None):
    if detectors==None:
        detectors = abs_3d.detectors(det_num=2,angles=[{'elevation':15,'azimuth':135},
                                                       {'elevation':15,'azimuth':225}])
    
    abs_series = {}

    for el in mass_atten:
        el_series = gpu_acf_series(density,mass_atten[el],intensity[el],angles,pix_size,detectors)
        
        abs_series[el] = el_series

    return abs_series