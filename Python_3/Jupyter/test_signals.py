#
# Python Test Signal Library
#
# Copyright (C) 2018 Wayne Mogg All rights reserved.
#
# This file may be used under the terms of the MIT License
#
# Author:		Wayne Mogg
# Date: 		March, 2018
# 
import numpy as np
import scipy.signal as sig

def make_random_signal(nsamp):
    """Make a single trace with random reflectivity

    A random reflectivity trace is convolved with a zero phase ricker wavelet
    Args:
        nsamp: the number of samples in the output trace
        
    Returns:
        A 1D array with the signal
    """
    ref = np.random.rand(nsamp)*2-1
    wav = sig.ricker(80,5)
    filtered = np.convolve(ref, wav,'same')
    return filtered

def make_delayed_signal_pair(nsamp, delay):
    """Make a pair of identical traces with specified delay

    A random reflectivity trace is convolved with a zero phase ricker wavelet
    and the created trace and a delayed version are returned
    Args:
        nsamp: the number of samples in the output trace
        delay: the number of samples to delay the second trace
        
    Returns:
        Two 1D arrays with the undelayed and delayed signal
    """
    ref = np.random.rand(nsamp+abs(delay))*2-1
    wav = sig.ricker(80,5)
    filtered = np.convolve(ref, wav,'same')
    if delay < 0 :
        return filtered[0:nsamp], filtered[-delay:nsamp-delay]
    else:
        return filtered[delay:nsamp+delay], filtered[0:nsamp]

class SphericalSignal(object):
    """Make a 3D spherical test signal
    
    Provides a 3D sinusoidal, hemisperical test signal and its spatial derivatives
    
    Args:
        factor: a parameter controlling the frequency content of the signal.
                Default is 5000.
        xsize: the size of the 3D signal in the 1st dimension. Default is 301.
        ysize: the size of the 3D signal in the 2nd dimension. Default is 301.
        zsize: the size of the 3D signal in the last dimension. Default is 301.
        deriv: what derivative of the test signal to create. Default is None. 
        
    """
    def __init__(self,factor=5000, xsize=301, ysize=301, zsize=301, deriv=None):
        self.xs = xsize
        self.ys = ysize
        self.zs = zsize
        self.factor = factor
        f0=.01
        k=.001
        xtmp = np.linspace(-xsize,xsize,xsize)
        ytmp = np.linspace(-ysize,ysize,ysize)
        ztmp = np.linspace(-2*zsize,0,zsize)
        self.x,self.y,self.z = np.meshgrid(xtmp,ytmp,ztmp, indexing='ij')
        t = (self.x**2+self.y**2+self.z**2)/factor
        if deriv == 'dx':
            self.data = 2/factor * self.x * np.cos(t)
        elif deriv == 'dy':
            self.data = 2/factor * self.y * np.cos(t)
        elif deriv == 'dz':
            self.data = 2/factor * self.z * np.cos(t)
        elif deriv == 'dxx':
            self.data = 2/factor * np.cos(t) - 4/(factor*factor) * np.square(self.x) * np.sin(t)
        elif deriv == 'dyy':
            self.data = 2/factor * np.cos(t) - 4/(factor*factor) * np.square(self.y) * np.sin(t)
        elif deriv == 'dzz':
            self.data = 2/factor * np.cos(t) - 4/(factor*factor) * np.square(self.z) * np.sin(t)
        elif deriv in ['dxy', 'dyx']:
            self.data = -4/(factor*factor) * self.x * self.y * np.sin(t)
        elif deriv in ['dxz', 'dzx']:
            self.data = -4/(factor*factor) * self.x * self.z * np.sin(t)
        elif deriv in ['dyz', 'dzy']:
            self.data = -4/(factor*factor) * self.y * self.z * np.sin(t)
        else:
            self.data = np.sin(t)

    def xSlice(self, x):
        """Return an y-z plane at location x
        
        Args:
            x: the x value of the required y-z plane
            
        Returns:
            A 2D array with the y-z plane if x is a valid index
            otherwise returns a plane of zeros.
        """
        if (x<=self.xs):
            return np.transpose(self.data[x,:,:])
        else:
            return np.zeros((self.data[0,:,:].shape))
        
    def ySlice(self, y):
        """Return an x-z plane at location y
        
        Args:
            y: the y value of the required x-z plane
            
        Returns:
            A 2D array with the x-z plane if y is a valid index
            otherwise returns a plane of zeros.
        """
        if (y<=self.ys):
            return np.transpose(self.data[:,y,:])
        else:
            return np.zeros((self.data[:,0,:].shape))

    def zSlice(self, z):
        """Return an x- plane at location z
        
        Args:
            z: the z value of the required x-y plane
            
        Returns:
            A 2D array with the x-y plane if z is a valid index
            otherwise returns a plane of zeros.
        """
        if (z<=self.zs):
            return self.data[:,:,z]
        else:
            return np.zeros((self.data[:,:,0].shape))

    def getXslice(self, x, xstep, ystep):
        """A generator for a series of data cubes along a y-z plane at location x
        
        Allows iteration along a y-z plane where at each interation a data cube
        of shape (2*xstep+1, 2*ystep+1, zsize) is returned. Cubes around the edge
        of the test signal volume are padded with the edge value.
        
        Args:
            x: the x value of the required y-z plane
            xstep: number of traces either side of the current location to include
            ystep: number of traces either side of the current location to indlude
            
        Returns:
            A series of data cubes along the specified y-z plane
        """
        tmp = np.pad( self.data, ((xstep,xstep),(ystep,ystep),(0,0)), mode='edge')
        for y in range(self.ys):
            yield tmp[x:x+2*xstep+1,y:y+2*ystep+1,:]

    def getYslice(self, y, xstep, ystep):
        """A generator for a series of data cubes along a x-z plane at location y
        
        Allows iteration along a x-z plane where at each interation a data cube
        of shape (2*xstep+1, 2*ystep+1, zsize) is returned. Cubes around the edge
        of the test signal volume are padded with the edge value.
        
        Args:
            y: the y value of the required x-z plane
            xstep: number of traces either side of the current location to include
            ystep: number of traces either side of the current location to indlude
            
        Returns:
            A series of data cubes along the specified x-z plane
        """
        tmp = np.pad( self.data, ((xstep,xstep),(ystep,ystep),(0,0)), mode='edge')
        for x in range(self.xs):
            yield tmp[x:x+2*xstep+1,y:y+2*ystep+1,:]

    def getZslice(self, z, xstep, ystep, zstep):
        """A generator for a series of data cubes on an x-y plane at location z
        
        Allows iteration over an x-y plane where at each interation a data cube
        of shape (2*xstep+1, 2*ystep+1, 2*zsize+1) is returned. Cubes around the edge
        of the test signal volume are padded with the edge value.The iteration
        proceeds along the xSlice direction.
        
        Args:
            z: the z value of the required x-y plane
            xstep: number of traces either side of the current location to include
            ystep: number of traces either side of the current location to indlude
            zstep: number of traces either side of the current location to indlude
            
        Returns:
            A series of data cubes on the specified x-y plane
        """
        tmp = np.pad( self.data, ((xstep,xstep),(ystep,ystep),(zstep,zstep)), mode='edge')
        for x in range(self.xs):
            for y in range(self.ys):
                yield tmp[x:x+2*xstep+1,y:y+2*ystep+1,z:z+2*zstep+1]

