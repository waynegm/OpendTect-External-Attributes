"""
Ported by Massimo Vassalli [http://mv.nanoscopy.eu massimo.vassalli@gmail.com]
"""
import numpy as np
def pwc_bilateral(y, soft=True, beta=200.0, width=5, display=True, stoptol=1e-3, maxiter=50):
# Performs PWC denoising of the input signal using hard or soft kernel
# bilateral filtering.
#
# Usage:
# x = pwc_bilateral(y, soft, beta, width, display, stoptol, maxiter)
#
# Input arguments:
# - y          Original signal to denoise of length N.
# - soft       Set this to 1 to use the soft Gaussian kernel, else uses
#              the hard kernel.
# - beta       Kernel parameter. If soft Gaussian kernel, then this is the
#              precision parameter. If hard kernel, this is the kernel
#              support.
# - width      Spatial kernel width W.
# - display    (Optional) Set to 0 to turn off progress display, 1 to turn
#              on. If not specifed, defaults to progress display on.
# - stoptol    (Optional) Precision of estimate as determined by square
#              magnitude of the change in the solution. If not specified,
#              defaults to 1e-3.
# - maxiter    (Optional) Maximum number of iterations. If not specified,
#              defaults to 50.
#
# Output arguments:
# - x          Denoised output signal.
#
# (c) Max Little, 2011. If you use this code for your research, please cite:
# M.A. Little, Nick S. Jones (2011)
# "Generalized Methods and Solvers for Noise Removal from Piecewise
# Constant Signals: Part I - Background Theory"
# Proceedings of the Royal Society A (in press)

    N = len(y)
    y = np.array(y)
    
    # Construct bilateral sequence kernel
    w = np.eye(N)
    for i in range(width):
        w=w+np.eye(N,k=i+1)+np.eye(N,k=-i-1)

    xold = y           # Initial guess using input signal
    d = np.zeros((N,N))
    
    if (display):
        if (soft):
            print('Soft kernel')
        else:
            print('Hard kernel')
        print('Kernel parameters beta={0}, W={1}'.format(beta, width))
        print('Iter# Change')
    
    # Iterate
    iiter = 1
    gap = np.Inf
    
    while (iiter < maxiter):
    
        if (display):
            print('{0} ; {1}'.format(iiter,gap))
        
        # Compute pairwise distances between all samples
        for i in range(N):
            d[:,i] = 0.5*(xold-xold[i])**2
        
        # Compute kernels
        if (soft):
            W = np.exp(-beta*d)*w    # Gaussian (soft) kernel
        else:
            W = np.zeros((N,N))
            W[np.where(d <= beta**2)] = w[np.where(d <= beta**2)]    # Characteristic (hard) kernel
        # Do kernel weighted mean shift update step
        xnew = np.sum(np.transpose(W)*xold,1)/np.sum(W,1)     
        gap = np.sum((xold - xnew)**2)

        # Check for convergence
        if (gap < stoptol):
            if (display):
                print('Converged in {0} iterations'.format(iiter))
            break
        
        xold = xnew
        iiter += 1
    
    if (display):
        if (iiter == maxiter):
            print('Maximum iterations exceeded\n')
    return xnew

if __name__ == "__main__":
    y = [1 ,1.1, 0.9, 1.1, 0.95, 2.1, 1.95, 2.0, 2.05, 3.11, 2.99, 3.05, 3.0]
    print('Perform test')
    x = pwc_bilateral(y,width=3)
    print(x)



