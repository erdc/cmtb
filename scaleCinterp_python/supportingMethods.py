# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 16:17:35 2014

@author: szthompson
"""
import numpy as np
import scipy as sp

def changeShape(dataObj, shape):
    """

    Args:
      dataObj: param shape:
      shape: 

    Returns:

    """
    if(np.shape(dataObj) != shape):
        reshapedDataObj = np.reshape(dataObj, shape)
        return reshapedDataObj
    else:
        return np.array(dataObj)
        
def omitNans(dataObj):
    """

    Args:
      dataObj: 

    Returns:

    """
    import math
    newDataObj = []
    for p in dataObj:
        foundNan = False
        if (math.isnan(p)):
            foundNan = True
        if(foundNan == False):
            newDataObj.append(p)
    newDataObj = np.array(newDataObj)
    return newDataObj

# DEPRICATED
#def constructMatrix(Xleft, Xmid, Xright):
#    N = np.size(Xleft)
#    n1, m1 = np.shape(Xmid)
#    n2, m2 = np.shape(Xright)
#    Q = []
#    for i in range(len(Xleft)):
#        Q.append(Xleft[i])
#    for i in range(len(Xmid)):
#        for j in range(len(Xmid[0])):
#            Q.append(Xmid[i,j])
#    for i in range(len(Xright)):
#        for j in range(len(Xright[0])):
#            Q.append(Xright[i,j])
#    R = np.reshape(Q, (N,1+m1+m2))
#    return R

# DEPRICATED   
#def multiplyMatrices(matrixA,matrixB):
#    result = np.zeros((len(matrixA),len(matrixB[0])),float)
#    for i in range(len(matrixA)): 
#       for j in range(len(matrixB[0])):
#           for k in range(len(matrixB)):
#               result[i][j] += matrixA[i][k] * matrixB[k][j]
#    return result

# DEPRICATED
#def findIndices(A, condition):
#    """
#    # taken from http://stackoverflow.com/questions/5957470/matlab-style-find-function-in-python
#    return [i for (i, val) in enumerate(A) if func(val)]
#    """   
#    B = A.flatten(1)
#    C = []
#    m = 0
#    """
#    if (condition == None):
#        for i in B:
#            if (i != 0 or i != np.nan):
#                C.append(m)            
#            m += 1
#    else:
#    """
#    for i in B:
#        if (eval(condition)):
#            C.append(m)            
#        m += 1
#    return np.array(C, dtype=int)
        
def makeWBFlow(Ylocal, Nysmooth, mindyg):
    """

    Args:
      Ylocal: param Nysmooth:
      mindyg: 
      Nysmooth: 

    Returns:

    """
    wbflow = np.exp(-((Ylocal - Ylocal[1]) / (1 * Nysmooth * mindyg)) ** 2)
    wbflow = wbflow + np.exp(-((Ylocal - Ylocal[-1]) / (1 * Nysmooth * mindyg)) ** 2);
    wbflow = 1 - wbflow
    wbflow = wbflow / max(wbflow[:])
    
    return wbflow
    
    # """ THE BELOW MAY NOT BE NEEDED
    # wb = 1-(Ei)/(wbfloor + Ei)
    # if(isflow > 0):
    #     # change the spline type
    #     #splinebc = [1,10] # different xshore,yshore bc
    #     # force to uniform
    #     #wb = wb.*(1-wbflow)
    #     wb = wb * wbflow
    #
    # from bspline import bspline_pertgrid
    # Zbs = bspline_pertgrid(Zi * (wbflow ** 2), wb, splinebcpert, splinelc, splinedxm)
    #
    # return wbflow
    # """
    
def consistentWeight(z, e2, wtol=0.1):
    """compute weights consistent with observations and expected error variance for each observation
    
    [w, s2] = consistentWeight(z, e2, wtol)
    
    Input:
    z, the observations
    e2, the error VARIANCE (SQUARED, DAMMIT!) at each observation
    wtol, (optional) convergence criteria (suggest 0.1 to 0.01)
    
    Output:
    w, the consistent weights
    s2, the estimated true variance
    
    assume (wj^2) = s2/(s2 + e2j), where s2 is true variance and e2j is error variance at observation j
    solve problem by guessing wj, then compute s2, then update wj
    assumes that true variance, s2, is constant over the data

    Args:
      z: param e2:
      wtol: Default value = 0.1)
      e2: 

    Returns:

    """
    # initial guess, weights are uniform
    #N = np.size(e2)
    winit = np.ones((z.shape[0],1), float)
    w = np.zeros((z.shape[0],1), float) # sum of weights
    cnt = 0

    while (cnt < 10 and np.max(np.abs(w - winit)) > wtol):
        cnt += 1
        # don't let large weights dominate too soon
        if(cnt > 1):
            winit = (w + winit) / 2
        # compute terms
        mu = np.sum(winit * z) / np.sum(winit) # weighted mean
        nu = np.size(z)
        # weighted mean squared residual-> true variance if weigths are accurate
        s2 = np.sum((winit * (z - mu))**2) / nu
        # when nu is small, this estimate ought to be replaced by the a priori error (at least)
        s2 = np.mean(((nu - 1) * s2 + e2) / nu)
        w = np.sqrt(s2 / (s2 + e2))
        # no further convergence if nu=1
        if(nu == 1):
            return w, s2
    return w, s2

def svd_invP(m, p):
    """invert matrix m
     retain eigen functions that account for p% of the variance
    
     m_inv = svd_invP(m, p);
    
     input
        m is a NOT necessarily a square matrix
        p is the percent variance to retain
    
     output
        m_inv is its inverse, in the SVD sense

    Args:
      m: param p:
      p: 

    Returns:

    """
    # the retained variance (does not need to be 90%)
    P = p / 100
    
    Nt, Nx = np.shape(m)
    
    # invoke svd
    from numpy.linalg import svd
    u, d, v = svd(m)
    
    # total variance
    D = np.diag(d)
    S = np.sum(D)
    
    # dd is used in the inverse calculation
    # if sum from 1 to i explains P*100% of variance, truncate
    dd = np.zeros((Nt,Nx), float)
    n = 0
    Df = D.flatten(1)
    sumD = 0
    for i in range(Nx-1):
        sumD += Df[i]
        if (sumD / S <= P):
            dd[i,i] = 1/d[i,i]
        else:
            n += 1
    # """
    # if (n>0):
    #     #print n,' un-used columns');
    # """
    # now do the inverse thing
    m_inv = v * (dd.conj().T) * (u.conj().T)#u.H)
    return m_inv

def si_wt(x, w=None):
    """[a] = si_wt(x, w)
    
    Input:
    x are nxm inputs, weights will be centered on x=0
    w are nx1 values of Root normalized signal/total variance of data
    
    Output:
    a are weights 0<=w<=1

    Args:
      x: param w:  (Default value = None)
      w:  (Default value = None)

    Returns:

    """
    n, m = np.shape(x)
    Ln = 2 # fix up so scaling sends weights to zero at x=1
    x = x * Ln
    if(w == None): # w=0 is the default, implying that no user input was recieved
        w = np.ones((n,1), float)
    else:
        w = (w + 1e-3) / (1 + 1e-3)
    # compute data-data covariance matrix (true variance at data points + noise along% diagonal)
    # assume true variance is constant at all locations (var=1)
    r = 0
    for i in range(1,m):
        tmp1 = np.tile(x[:,i].conj().T, (n,1))
        tmp2 = np.tile(x[:,i], (1,n)) 
        tmp2 = np.reshape(tmp2, np.shape(tmp1)) # tmp2 is of an incompabitible shape with tmp1
        r = r + (tmp1 - tmp2)**2
    r = np.exp(-r) * (w * (w.conj().T)) # has covariance of w.^2 along diagonal
    # diagonal elements have observation error added
    #r = r + diag(1-w) # not quite right, if w=0, get cov of 2 on diagnal. should be one, and zero everywhere else
    r = r + np.diag(1-w**2) # now, if r is scaled correctly, this oughta work
    
    # invert it
    P = np.ceil(95 + 5 * (100 / (100 + n)))
    #P=100 # only reduce P if many data
    
    from .supportingMethods import svd_invP
    r = svd_invP(r, P) # approach 95% for large N
    #r = np.inv(r)
    
    # now, data-true covariance vector
    ri = np.exp(-(x**2)*np.ones(m,float,1))*(w**2)
    
    # calculate the weights
    w = np.dot(r, ri)
    return w

def hanning_wt(r):
    """w = hanning_wt(x)
    
    Input:
    x are nxm inputs, weights will be centered on x=0
    
    Output:
    w are weights 0<=w<=1

    Args:
      r: 

    Returns:

    """
    m = np.size(r)    
        
    # convert to radial distance
    tmp1 = r**2
    tmp1 = np.reshape(tmp1, (len(tmp1), 1)) # go from shape (len(r**2),) to (len(r**2),1) for compatibility 
    r = np.sqrt(tmp1 * np.ones((m,1), float))#r = sqrt((r.^2)*ones(m,1));
    
    # hanning window
    r = (1 - np.cos(np.pi * (0.5 + 0.5 * r)) ** 2) * (r <= 1)
    
    return r

def loess_wt(r):
    """[w] = loess_wt(x)
    Input: x are nxm inputs, weights will be centered on x=0
    Output: w are weights 0<=w<=1

    Args:
      r: 

    Returns:

    """
    
    n, m = np.shape(r)
    
    # convert to radial-squared distance
    r = np.dot((r**2), np.ones((m,1), float))
    
    # the loess weighting function (zero if x>1)
    r = ((1 - (r**3))**3) * (r < 1)  # used by greenslade(must be typo)
    
    #r = ((1-(sqrt(r)**3))**3)*(r<1)  # used by schlax and chelton
    
    # x = ((1-(x**1.5))**3)*(x<1) # used by schlax and chelton
    return r
    
def loess_kernelND(d, p, Dx=0.05):
    """

    Args:
      d: param p:
      Dx: Default value = 0.05)
      p: 

    Returns:

    """
    x = np.arange(-(1 + Dx), (1 + Dx), Dx) 
    x = np.reshape(x, (np.size(x, axis=0), 1)) # reshape x to be a column vector 
    N = np.size(x, axis=0)
    
    # build ND input
    if (d == 1):
        X = x
    elif (d == 2):
        X, Y = np.meshgrid(x, x)
        X = np.array([X.T.flatten(1), Y.T.flatten(1)]).T
    elif (d == 3):
        X, Y, T = np.meshgrid(x, x, x)
        X = np.array([T.T.flatten(1), X.T.flatten(1), Y.T.flatten(1)]).T
    
    # get weights
    from .supportingMethods import loess_wt
    W = loess_wt(X)
    
    # get basis for radially symmetric output
    r = np.concatenate((x, np.zeros((N,d-1), float)), axis=1)
    w = loess_wt(r)
    
    # build quadratic input
    m = d + 1
    n = N**d
    
    if(p == 1):
       # build linear input
       X = np.concatenate((np.ones((n,1), float), X), axis=1)
       r = np.concatenate((np.ones((N,1), float), r), axis=1)
    elif(p == 2):
       # get quadratic terms
       # there will be (m^2 + m)/2 entries (including the constant)
    
       q = 0.5 * m * (m+1)
       X = np.concatenate((np.ones((n,1), float), X, np.zeros((n,q-m), float)), axis=1)
       r = np.concatenate((np.ones((N,1), float), r, np.zeros((N,q-m), float)), axis=1)
    
       for i in range(1, d+1):
           # get the quadratic self-terms
           for j in range(1, i+1):
               m  += 1
               X[:,m-1] = X[:,i] * X[:,j]
               r[:,m-1] = r[:,i] * r[:,j]
    
    # weight all components
    Xw = X * np.kron(np.ones((1,m), float), W)
    r = r * np.kron(np.ones((1,m), float), w)
    
    # get the scaling matrix-- solves linear regression
    Xx = np.dot(Xw.conj().T, Xw) / (N**d)  # Use numpy.dot to do matrix multiplication
    from numpy.linalg import inv
    Xx_inv = inv(Xx)
    
    # we can get radially symmetric result now
    # and this is the thing that is mult against the weighted data
    tmp = np.dot(r, Xx_inv[:,0])
    tmp = np.reshape(tmp, (np.size(tmp, axis=0), 1))
    ar = tmp * (w / N) # Use numpy dot product to do matrix multiplication
    r = x 
    return r, ar
    # """
    # # or continue to see the beast in N-dimensional space
    # # and this is the thing that is mult against the weighted data
    # bX = np.dot(Xw, Xx_inv[:,0])
    # bX = np.reshape(bX, (np.size(bX, axis=0), 1))
    #
    # # so put the weight in this thing, rather than data, to get the indicator
    # # function
    # tmp1 = np.reshape(bX[:,0], (np.size(bX[:,0], axis=0), 1))
    # a = tmp1 * W
    #
    # # and reshape
    # if (d == 1):
    #     A = a
    # else:
    #     A = np.reshape(a, (N, N))#np.shape(N * np.ones((1,d), float)))
    #
    # # here is the 3-d surface in 1-d
    # ar = A[:, (N+1)/2, (N+1)/2]
    #
    # #return r, ar
    # """
    
def regr_xzw(X, z, w=None, nargout=2):    
    """[b,brmse,sk,n,msz,msr,nmse] = regr_xzw(X,z,w);
    
     general linear regression call
     -nan- returned if data-data correlation matrix is badly conditioned
    
     Input
       X, nxm dependendant variables, comprised of dataX and dataY as column vector entries
       z, nx1 observations
       OPTIONAL w, nx1 weights (0 means observation has no influence)
    
     Output
       b, mx1 estimated parameters: z^ = X*b;
       brmse, mx1 estimated variances (root mean square error) of parameter(s)
          (confidence intervals assume gaussian white-noise dist., with bmse estimated variance)
       sk, the model skill
       n, the effective dof =(sum(w)/max(w))
       msz, variance of data
       msr, variance of residuals
       nmse, percent of white error input variance passed by weights

    Args:
      X: param z:
      w: Default value = None)
      nargout: Default value = 2)
      z: 

    Returns:

    """
    w = None
    
    # inputs
    n, m = np.shape(X)
    nz = np.size(z)
    if w == None: # Defualt input 
        w = np.ones((n,1), float)
        nw = n
    else: # User-overwrite input
        nw = np.size(w, axis=0)
       
    # init output
    b = np.nan * np.ones((m,1), float)
    brmse = b
    sk = np.nan
    msz = np.nan
    msr = np.nan
    nmse = 1
    
    if(nz != n or  nw != n or nw != nz):
        print('X and z or w are different lengths \n')
        return b, brmse#, sk, n, msz, msr, nmse
    
    # find valid data by summing
    tmp = np.concatenate((X, z, w), axis=1)
    idd = (np.nonzero(np.isfinite(np.dot(tmp, np.ones((m+2,1), float)))==1))[0]
    if (np.size(idd) < 2):
        print('n < 2 -- exiting\n')
        return b, brmse #, sk, n, msz, msr, nmse
    
    # number of dof
    n = np.sum(w[idd]) / max(w[idd])
    #n = n[0] # some wierd dimensionality thing happens... you gotta extract n out of the structure that results from the above
    
    # convert to weighted space (priestly p.315)
    # Fienen pointed out this is wrong: z = (z).*w; X = X.*(repmat(w,1,m));
    try:
        Q = sp.sparse(np.diag(w[idd]**2))
    except:
        print('Q is too big, use constant!')
        Q = 1
    
    # and compute covariances
    # wrong: XX = (X(id,:)'*X(id,:))/n; # wrong: XZ = (z(id)'*X(id,:))/n;
    #tmp1 = X[idd,:].conj().T
    #tmp2 = Q * X[idd,:]
    Xx = np.dot(X[idd,:].conj().T, Q * X[idd,:])/n
    Xz = np.dot(z[idd].conj().T, Q*X[idd,:])/n
    
    # solve the weighted least squares problem
    from numpy.linalg import inv        
    XX_inv = inv(Xx)
    if (nargout == 2):
        return b, brmse
        
    # compute parameters
    b = np.dot(XX_inv, Xz.conj().T); 
    
    # model residuals
    msz = np.dot(z[idd].conj().T, Q*z[idd])/n
    msz = msz[0,0] 
    msr = msz - np.dot(np.dot(b.conj().T, Xx), b)
    msr = msr[0,0]
    sk = 1 - msr/msz
    
    # and perhaps we want all variance estimates
    # mse = XX_inv(1)*msr/(n-m)
    brmse = np.sqrt(np.diag(XX_inv) * msr / (n - m))
    
    # get normalized error, based on convoltion
    if (nargout==7):
        # first comput regresion weights, assuming first input is all ones
        tmp = np.reshape(XX_inv[:,1], (len(XX_inv[:,1]),1))
        bX = np.dot(X[idd,:], tmp)
        a = bX * w[idd] # element-wise multiplication
        a = a / sum(a)
        # sum of squared weights is normalized error: also, good est of dof
        nmse = np.dot(a.conj().T, a)
    
    return b, brmse, sk, n, msz, msr, nmse
    # """
    # Notes
    #  get the right error estimate
    #  read priestly, page 368: mse/msr ~ chi-sq(m)/(N-m)
    #  here is argument:
    #  var_observed = var_model + var_residual = var_true + var_noise
    #  and
    #  var_model = var_true + var_artificial
    #     var_true is variance of perfect model
    #     var_noise is additive white noise
    #     var_artificial is due to random correlations, i.e., aliased
    #  a linear model will pick up this much of noise
    #  var_a = (m/N) var_n, our extra bit of information
    #  Thus,
    #  var_n = var_r + var_a = var_r + (m/N) var_n = var_r /(1-m/N)
    #        = var_r*N/(N-m), expected value
    #  and
    #  var_a = var_r (m/N)/(1-m/N)
    #        = (m/(N-m)) var_r
    #  thus: mse_of_model = ( msr*m/(N-m) );
    #  this is plausible error in model, assumed uniform over range of data
    #  BUT, we are interested in error of estimate of b(1), the value at x=0
    #  See priestly page on regression models, which states that
    #  b are normally dist. around b_true, with var(b) = var_noise*diag(XX_inv)
    #  msn = msr*N/(N-m);
    #  mse_of_b(1) = XX_inv(1)*msn/N; -- Look to the F-dist to explain ratio of variances
    #  after testing synthetic examples, conclude that this is robust
    #  even leaving large mean values in!
    #
    #  note on weighted least squares
    #  replace N with sumW where sumW= sum of weights and weights ~ 1/sigma_noise
    #  for Q to insert between all the cross-correlations
    #
    #  test
    #  for j=1:100
    #      N=10; X=[ones(N,1),[1:N]']; z = 1*randn(N,1)+[1:N]'; w=ones(N,1); %w(1:5)=0.5;
    #      [b,brmse,sk,n,msz,msr,nmse] = regr_xzw(X,z,w);
    #      S(j).sk = sk;
    #      S(j).brmse=brmse';
    #      S(j).b=b';
    #  end
    #  actual parameter error
    #   mean(cat(1,S.b)) =  0.0059    1.0080
    #   std(cat(1,S.b))  =  0.6699    0.1069
    #  predicted parameter errors
    #   mean(cat(1,S.brmse))  =    0.6467    0.1042
    #  variance of parameter errors (just to show that it is relatively small)
    #   std(cat(1,S.brmse)) =      0.1768    0.0285
    # """





