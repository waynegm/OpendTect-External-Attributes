"""
Ported by Massimo Vassalli [http://mv.nanoscopy.eu massimo.vassalli@gmail.com]
"""
import scipy.sparse as sparse
import scipy.sparse.linalg as splin
import numpy.linalg as linalg
import numpy as np

def pwc_tvdip(y, lamb=[1.0], display=True, stoptol=1e-3, maxiter=60, full=False):
    # Performs discrete total variation denoising (TVD) using a primal-dual
    # interior-point solver. It minimizes the following discrete functional:
    #
    #  E=(1/2)||y-x||_2^2+lambda*||Dx||_1,
    #
    # over the variable x, given the input signal y, according to each
    # value of the regularization parameter lambda > 0. D is the first
    # difference matrix. Uses hot-restarts from each value of lambda to speed
    # up convergence for subsequent values: best use of this feature is made by
    # ensuring that the chosen lambda values are close to each other.
    #
    # Usage:
    # [x, E, s, lambdamax] = pwc_tvdip(y, lambda, display, stoptol, maxiter)
    #
    # Input arguments:
    # - y          Original signal to denoise, size N x 1.
    # - lambda     A vector of positive regularization parameters, size L x 1.
    #              TVD will be applied to each value in the vector.
    # - display    (Optional) Set to 0 to turn off progress display, 1 to turn
    #              on. If not specifed, defaults to progress display on.
    # - stoptol    (Optional) Precision as determined by duality gap tolerance,
    #              if not specified, defaults to 1e-3.
    # - maxiter    (Optional) Maximum interior-point iterations, if not
    #              specified defaults to 60.
    #
    # Output arguments:
    # - x          Denoised output signal for each value of lambda, size N x L.
    # - E          Objective functional at minimum for each lambda, size L x 1.
    # - s          Optimization result, 1 = solved, 0 = maximum iterations
    #              exceeded before reaching duality gap tolerance, size L x 1.
    # - lambdamax  Maximum value of lambda for the given y. If
    #              lambda >= lambdamax, the output is the trivial constant
    #              solution x = mean(y).
    #
    # (c) Max Little, 2011. Based around code originally written by 
    # S.J. Kim, K. Koh, S. Boyd and D. Gorinevsky. If you use this code for
    # your research, please cite:
    # M.A. Little, Nick S. Jones (2011)
    # "Generalized Methods and Solvers for Noise Removal from Piecewise
    # Constant Signals: Part I - Background Theory"
    # Proceedings of the Royal Society A (in press)
    #
    # This code is released under the terms of GNU General Public License as
    # published by the Free Software Foundation; version 2 or later.

    y = np.array(y)
    

    # Search tuning parameters
    ALPHA     = 0.01   # Backtracking linesearch parameter (0,0.5]
    BETA      = 0.5    # Backtracking linesearch parameter (0,1)
    MAXLSITER = 20     # Max iterations of backtracking linesearch
    MU        = 2      # t update
    
    N = len(y)    # Length of input signal y
    M = N-1          # Size of Dx

    # Construct sparse operator matrices 
    
    O1 = sparse.lil_matrix((M,M+1))
    O2 = sparse.lil_matrix((M,M+1))
    for i in range(M):
        O1[i,i]=1.0
        O2[i,i+1]=1.0
    D = O1-O2
    
    DDT = D.dot(D.transpose())
    Dy  = D.dot(y)
    
    # Find max value of lambda
    lambdamax = np.max(np.abs(splin.spsolve(DDT,Dy)))
    
    if (display):
        print('lambda_max={0}'.format(lambdamax))

    L = len(lamb)
    x = np.zeros((N, L))
    s = np.zeros(L)
    E = np.zeros(L)
    
    # Optimization variables set up once at the start
    z    = np.zeros(M)   # Dual variable
    mu1  = np.ones(M)   # Dual of dual variable
    mu2  = np.ones(M)   # Dual of dual variable
    
    # Work through each value of lambda, with hot-restart on optimization
    # variables
    for l in range(L):
        t    =  1e-10; 
        step =  np.Inf;
        f1   =  z-lamb[l];
        f2   = -z-lamb[l];
    
        # Main optimization loop
        s[l] = True;
    
        if (display):
            print('Solving for lambda={0}, lambda/lambda_max={1}\nIter# Primal    Dual      Gap'.format(lamb[l], lamb[l]/lambdamax))
    
        for iters in range(maxiter):
            DTz = (z*D)
            DDTz = D*DTz 
            w    = Dy-(mu1-mu2)
            
            # Calculate objectives and primal-dual gap
            pobj1 = 0.5*w.dot(splin.spsolve(DDT,w))+lamb[l]*np.sum(mu1+mu2)
            pobj2 = 0.5*DTz.dot(DTz)+lamb[l]*np.sum(np.abs(Dy-DDTz))
            pobj = min(pobj1,pobj2)
            dobj = -0.5*DTz.dot(DTz)+Dy.dot(z)
            gap  = pobj - dobj
            
            if (display):
                print('{0} -- {1} -- {2} -- {3}'.format(iters, pobj, dobj, gap))

            if (gap <= stoptol):
              s[l] = True
              break;
    
            if (step >= 0.2):
                t = max(2*M*MU/gap, 1.2*t)
    
            # Do Newton step
            rz      =  DDTz - w          
            SSS = sparse.lil_matrix((M,M))
            for i in range(M):
                SSS[i,i]=(mu1/f1+mu2/f2)[i]
            S       =  DDT - SSS
            r       = -DDTz + Dy + (1/t)/f1 - (1/t)/f2
            dz=splin.spsolve(S,r)
            dmu1    = -(mu1+((1/t)+dz*mu1)/f1)
            dmu2    = -(mu2+((1/t)-dz*mu2)/f2)
          
            resDual = rz
            resCent = resCent = np.concatenate((-mu1*f1-1/t, -mu2*f2-1/t))
            residual= residual=np.concatenate((resDual,resCent))
            negIdx2 = (dmu2 < 0)
            negIdx1 = (dmu1 < 0)
            
            step=1.0
            if (negIdx1.any()):
                print()
                step = min( step, 0.99*min(-mu1[negIdx1]/dmu1[negIdx1]) )
            if (negIdx2.any()):
                step = min( step, 0.99*min(-mu2[negIdx2]/dmu2[negIdx2]) )
            
            for liter in range(MAXLSITER):
                newz    =  z  + step*dz
                newmu1  =  mu1 + step*dmu1
                newmu2  =  mu2 + step*dmu2
                newf1   =  newz - lamb[l]
                newf2   = -newz - lamb[l]
                
                newResDual  = DDT*newz - Dy + newmu1 - newmu2
                newResCent  = np.concatenate( (-newmu1*newf1-1/t,-newmu2*newf2-1/t) )
                newResidual = np.concatenate( (newResDual,newResCent) )
                
                if ( (max(max(newf1),max(newf2)) < 0) and (linalg.norm(newResidual) <= (1-ALPHA*step)*linalg.norm(residual)) ):
                    break
                step = BETA*step
                
            z = newz
            mu1 = newmu1
            mu2 = newmu2
            f1 = newf1
            f2 = newf2
            
        x[:,l] = y-np.transpose(D)*z
        E[l] = 0.5*sum((y-x[:,l])**2)+lamb[l]*sum(abs(D*x[:,l]))
        
        # We may have a close solution that does not satisfy the duality gap
        if (iters >= maxiter):
            s[l] = False
        if (display):
            if (s[l]==True):
                print('Solved to precision of duality gap {0}'.format( gap))
            else:
                print('Max iterations exceeded - solution may be inaccurate')
    if full:
        return x, E, s, lambdamax
    return x

if __name__ == "__main__":
    y = [1 ,1.1, 0.9, 1.1, 0.95, 2.1, 1.95, 2.0, 2.05, 3.11, 2.99, 3.05, 3.0]
    print('Perform test')
    x = pwc_tvdip(y,[1.0])
    print(x)
