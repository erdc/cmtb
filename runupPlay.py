import numpy as np
import matplotlib.pyplot as plt
import pickle

#load existing pickle file
pickleSaveFname = '/Users/l6kim/cmtb/data/SWASH/base/2015-11-05T000000Z/2015-11-05T000000Z_io.pickle'
with open(pickleSaveFname, 'rb') as fid:
    SWIO = pickle.load(fid)

#load existing mat file
matFile = '/Users/l6kim/cmtb/data/SWASH/base/2015-11-05T000000Z/20151105T000000Z.mat'
simData, simMeta = SWIO.loadSwash_Mat(fname=matFile)

eta = simData['eta'].squeeze()

#now using Chuan's runup code
r_depth = 0.08 #4.0 * np.nanmax(np.abs(h[runupInd][1:] - h[runupInd][:-1]))

# Preallocate runup variable
runup = np.zeros(eta.shape[0])
x_runup = np.zeros_like(runup)

for aa in range(runup.shape[0]):
    # Water depth
    wdepth = eta[aa, :] + simData['elevation']
    # Find the runup contour (search from left to right)
    wdepth_ind = np.argmin(abs(wdepth - r_depth)) #changed from Chuan's original code
    # Store the water surface elevation in matrix
    runup[aa] = eta[aa, wdepth_ind]  # unrealistic values for large r_depth
    # runup[aa]= -h[wdepth_ind]
    # Store runup position
    x_runup[aa] = simData['xFRF'][wdepth_ind]


plt.plot(np.arange(0, 4097), runup)


