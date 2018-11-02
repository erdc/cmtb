# -*- coding: utf-8 -*-
"""
Created on Wed Nov 05 20:31:07 2014

@author: jwlong
"""
import numpy as np
from .supportingMethods import consistentWeight

def subsampleData(X, z, e, Dx):
    """interpolate data into regular sample bins using boxcar window

     Args:
         X: an NxM set of coordinates
         z: an Nxr array of observations, where the 2:rth columns are averaged identically to the first column
         e: an Nx1 array of errors (used to weight sum) (optional) ONLY APPLIES TO z(:,1), other variables (if exist)
             weighted by these values
         DX: an MxN2 array of scale parameters, indicating the step size in each dimension
         N2: is either 1 (indicating constant smoothing lengthscales, or M, indicating variable lengthscales
             for each datapoint
    
     Returns
       Xi: the mean position of the data in each cell
       zi: the mean value at each interp. cell
       si: the standard error (=std. dev./sqrt(n)) (or, if ni<3, insert average value of all other si values)
           ONLY COMPUTED FOR z(:,1), others are weigthed identically
       ni: the number of observations going into each cell
       Ji: the array of indices into each cell
       Jmax: the array of number of cells in each dimension
       X0: the location of the first grid point (i.e., location at Ji=1)

     Notes:
        to make a quick grid of the data:
            [XX,YY] = meshgrid([1:Jmax(1)]*DX(1)+X0(1),[1:Jmax(2)]*DX(2)+X0(2));
            ZZ = repmat(nan,Jmax(1),Jmax(2)); % careful, read in flipped
            EE = ZZ;
            ZZ(Ji) = zi; ZZ = ZZ'; % flip to usual orientation for matlab
            EE(Ji) = si; EE = EE';
            pcolor(XX,YY,ZZ);

    """
        # allow nans
    # 03 Jan 2011: ngp adds "+e+ so that data with nan error is chucked
    #np.seterr(invalid='ignore')
    tmp = z + e
    X = X[~np.isnan(tmp).any(1),:]
    z = z[~np.isnan(tmp)]
    e = e[~np.isnan(tmp)]
    
    # Modify input for compatibility
    z = np.reshape(z, (len(z),1))    
    
    # find out size of z (might have multiple columns
    N, r = np.shape(z)
    
    N, M = np.shape(X)
    # check to see if data errors were passed in
    #if(Dx == None):
        # need to pass e to DX and generate e
    #    Dx=e
    #    e = np.ones((N,1), float)
    
    # what if constant error passed in
    if(len(e) == 1):
        # replicate and square it
        e = np.zeros((N,1), float) + e**2
    else:
        # just square it
        e = e**2
    
    e = e[:, np.newaxis]
    
    # Turn error into a weight
    wtol = 0.1  # close enough
    tmp = z[:, 0]
    tmp = tmp[:, np.newaxis]
    w, _ = consistentWeight(tmp, e, wtol)

    # map data to scaled points
    # J = 1,1...,1 is location X0(1,1,...,1)
    X0 = np.floor(X.min(0) / Dx) * Dx # make nice integer values
    J = np.round(1 + (X - np.tile(X0, (N,1))) / np.tile(Dx,(N,1)))
    
    # map these to index into multi-D array of unique indices
    _, Jm = np.shape(J)
    tmp = []
    for i in range(0, Jm): # Create Jmax, the vector comprised of the max values of the columns of J
        tmp.append(np.max(J[:,i]))

    Jmax = np.array(tmp)
    del tmp #, Jn # Jn was never actually needed
    Ji = np.ones((N,1), float)
    Jprod = 1
    for i in range(0,M):
        tmp = np.reshape(J[:,i], (len(J[:,i]),1))
        Ji = Ji + (tmp - 1) * Jprod
        del tmp
        Jprod = Jprod * Jmax[i]
    
    # sort them
    #[Jisort, sortid] = sort(Ji) ##### MATLAB ORIGINAL #####
    sortid = np.argsort(Ji, axis=0).flatten()
    Jisort = Ji[sortid]
    w = w[sortid]
    e = e[sortid]
    z = z[sortid]
    X = X[sortid,:]
    
    # how many unique?
    Ni = np.unique(Jisort)
    # Ni = np.sum(np.diff(Jisort, axis=0) > 0) + 1
    
    # initialize output arrays that are as large as the largest index
    zi = np.zeros((Ni.shape[0],1), dtype=float) #np.tile(0,(Ni,r)) # holds weighted sum, initially
    ni = np.zeros((Ni.shape[0],1), dtype=float) # number of observations
    wi = np.zeros((Ni.shape[0],1),dtype= float) # sum of weights
    w2zi = np.zeros((Ni.shape[0],1), dtype=float) # weights against data
    w2ei = np.zeros((Ni.shape[0],1), dtype=float) # weights against a priori errors
    w2i = np.zeros((Ni.shape[0],1), dtype=float) # sum of squared weights
    si = np.zeros((Ni.shape[0],1), dtype=float) # holds weighted sum of squares, initially
    Ji = np.zeros((Ni.shape[0],1), dtype=float)
    Xi = np.zeros((Ni.shape[0],M), dtype=float)#np.tile(0,(Ni,M))
    
    # scan through observations
    for i in range(Ni.shape[0]):
        # print '%.2f Percent Complete ' % (float(i)/Ni.shape[0])
        ind = np.where(Jisort == Ni[i])[0]
        ni[i] = len(ind)
        wind = w[ind]  # calculating these things only ones
        wind2 = wind ** 2

        wi[i] = sum(wind)
        w2i[i] = sum(wind2)
        zi[i] = sum(z[ind] * wind)
        w2zi[i] = sum(z[ind] * wind2)
        si[i] = sum((z[ind] * wind)**2)
        w2ei[i] = sum(e[ind] * wind2)
        Xi[i,:] = sum(X[ind,:] * wind)
        # del ind
        
#    cnt = 0
#    Ji[0] = Jisort[0]
#    print 'Subsampling data'
#    for i in range(0,N):
        # update the multi-d index and count
#        if(Jisort[i] > Ji[cnt]):
#            cnt += 1
#            Ji[cnt] = Jisort[i]
            
        # sum values for this index
#        if (np.isfinite(z[int(sortid[i]),0])):
#            ni[cnt] += 1
#            wi[cnt] += w[sortid[i]]
#            w2i[cnt] += w[sortid[i]]**2
#            zi[cnt,:][0] += z[sortid[i],:][0] * w[sortid[i]]
#            w2zi[cnt] += z[sortid[i],0] * (w[sortid[i]]**2)
#            si[cnt] += (z[sortid[i],0] * w[sortid[i]])**2
#            w2ei[cnt] += e[sortid[i]] * (w[sortid[i]]**2)
            
            # keep track of weighted mean data location within cell
#            Xi[cnt,:] += X[sortid[i],:] * w[sortid[i]]
                
    # eliminate cases with missing data (e.g., nans)
    idd = np.where(ni.flatten() > 0) # ni is 1-d... but must still be flattened in order for idd to be of the appropriate dimensions
    ni = ni[idd]
    
    # means
    zi = (zi[idd] / np.tile(wi[idd], (1,r))).T
    Xi = Xi[idd] / np.tile(wi[idd], (1,M)) # use this for mean data location in cell
    
    # mean square residual
    if (r == 1): # Construct the polynomial terms for si based on the shape of zi 
        A = np.expand_dims(zi.flatten() * w2zi[idd].flatten(), axis=1)
        B = np.expand_dims(w2i[idd].flatten(1) * (zi**2).flatten(1), axis=1)
    else:
        A = np.expand_dims(zi[:,0].flatten() * w2zi[idd].flatten(), axis=1)
        B = np.expand_dims(w2i[idd].flatten(1) * (zi[:,0]**2).flatten(1), axis=1)
    si = (si[idd] - 2 * A  + B) / ni
    del A, B
    
    # return as mean square error (fraction of residual that passes through)
    si = si / (1 + ni)
    
    # play usual trick to deal with cases where ni is smalln 
    # weight computed residual by ni-1, and the a priori error variance by 1, sum of weights = ni
    # and return as square root
    si = np.sqrt( ((ni-1) * si + w2ei[idd] / w2i[idd]) /ni )
    
    # finally, return indices into the virtual n-dimensional grid
    Ji = Ji[idd]  
    
    return Xi, zi.T, si