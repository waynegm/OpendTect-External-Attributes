"""
Ported by Massimo Vassalli [http://mv.nanoscopy.eu massimo.vassalli@gmail.com]
"""

import numpy as np
from numpy.matlib import repmat
from numpy.random import randint

def randsample(y,K):
    return y[randint(0,len(y)-1,K)]

def pwc_cluster(y,K=None,soft=False,beta=0,biased=False,display=True,stoptol=1e-5,maxiter=50):
    # Performs PWC denoising of the input signal using hard or soft mean-shift,
    # K-means, or likelihood mean shift clustering.
    #
    # Usage:
    # x = pwc_cluster(y, K, soft, beta, biased, display, stoptol, maxiter)
    #
    # Input arguments:
    # - y          Original signal to denoise of length N.
    # - K          Number of PWC levels (clusters). If K<N, performs K-means
    #              clustering. Choose K=[] or K=N to perform mean-shift.
    # - soft       Set this to True to use the soft Gaussian kernel, else uses
    #              the hard kernel.
    # - beta       Kernel parameter. If soft Gaussian kernel, then this is the
    #              precision parameter. If hard kernel, this is the kernel
    #              support.
    # - biased     Set this to True to use 'biased' mode: that is, the weighted
    #              mean of the input samples, rather than the current estimated
    #              PWC signal. Note that if performing K-means, this is
    #              automatically True.
    # - display    (Optional) Set to False to turn off progress display, True to turn
    #              on. If not specifed, defaults to progress display on.
    # - stoptol    (Optional) Precision of estimate as determined by square
    #              magnitude of the change in the solution. If not specified,
    #              defaults to 1e-5.
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
    y = np.array(y)
    N = len(y)    
    if K==None:
        K = N
        if (display):
            print('Mean-shift mode');
    else:
        if (display):
            print('K-means mode K={0}'.format(K));    
    
    xold = y;
    if (K < N):
        biased = True;
        xold = randsample(y,K)     # Random cluster centroids
    else:
        xold = y                   # Initialise to input signal
    
    if (display):
        if (soft):
            print('Soft kernel')
        else:
            print('Hard kernel')
        print('Kernel parameter beta={0}'.format(beta))
        if (biased):
            print('Biased (likelihood) mode')
        else:
            print('Unbiased mode')
        print('Iter# Change')
    
    d = np.zeros((N,K))
    
    if (K < N):
        I = np.eye(K)                 # Indicators for cluster centroids
    
    # Iterate
    iter = 1
    gap = np.Inf
    while (iter < maxiter):
    
        if (display):
            print('{0} {1}'.format(iter,gap))
    
        # Compute pairwise distances
        if (K < N):
            # Distances between cluster centroids and input samples
            for i in range(K):
                d[:,i] = 0.5*(y-xold[i])**2
        else:
            # Distances between current estimated PWC samples
            for i in range(N):
                d[:,i] = 0.5*(xold-xold[i])**2
        
        # Compute kernels
        if (soft):
            # Soft Gaussian kernel
            W = np.exp(-beta*d)
            if (K < N):
                # Soft cluster assignment kernel
                h = repmat(np.sum(W,1),1,K)
                W = W/np.transpose(h.reshape(K,N))
        else:
            if (K == N):
                # Hard characteristic kernel
                W = np.zeros((N,K))
                W[np.where(d<=(beta**2))]=1
            else:
                # Hard cluster indicator kernel
                W = I[np.argmin(d,1),:]
        
        # Normalize kernel to find mean weights
        w = 1/np.sum(W,1)
        
        #Kernel weighted update step
        den = np.dot(np.transpose(W),w)
        if (biased):
            num = np.dot(np.transpose(W),(w*y))
            #xnew = (W'*(w.*y))./(W'*w);
        else:
            num = np.dot(np.transpose(W),(w*xold))
            #xnew = (W'*(w.*xold))./(W'*w);
        xnew= num/den
    
        gap = np.sum((xold - xnew)**2)
    
        # Check for convergence
        if (gap < stoptol):
            if (display):
                print('Converged in {0} iterations'.format(iter))
            break;

        xold = xnew
        iter += 1
    
    if (display):
        if (iter == maxiter):
            print('Maximum iterations exceeded')
    
    if (K < N):
        # Assign samples to nearest cluster centroids when K < N
        for i in range(K):
            d[:,i] = np.abs(y-xnew[i])**2;
        x = xnew[np.argmin(d,1)]
    else:
        x = xnew
    return x    
    
if __name__ == "__main__":
    y = [1 ,1.1, 0.9, 1.1, 0.95, 2.1, 1.95, 2.0, 2.05, 3.11, 2.99, 3.05, 3.0]
    print('Perform test')
    x = pwc_cluster(y,5)
    print(x)
