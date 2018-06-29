from testbedutils import sblib as sb
import netCDF4 as nc
from matplotlib import pyplot as plt
import numpy as np
"""
This code was designed to check consistancy at the offshore boundary of the inner nested domain.
"""
CBncfile = nc.Dataset('http://134.164.129.62:8080/thredds/dodsC/CMTB/waveModels/STWAVE/CB/Local-Field/Local-Field.ncml') #'http://crunchy:8080/thredds/dodsC/CMTB/waveModels/STWAVE/CB/Local-Field/Local_Field.ncml')
HPnc = nc.Dataset('http://134.164.129.62:8080/thredds/dodsC/CMTB/waveModels/STWAVE/CBHP/Local-Field/Local-Field.ncml')
HPncStatic = nc.Dataset('http://134.164.129.62:8080/thredds/dodsC/CMTB/projects/STWAVE/CBHP/SingleBathy_2013-12-06/Local-Field/Local-Field.ncml')#'http://crunchy:8080/thredds/dodsC/CMTB/projects/waveModels/STWAVE/CBHP/SingleBathy_2013-12-06/Local-Field/Local-Field.ncml')
CBTncfile = nc.Dataset('http://crunchy:8080/thredds/dodsC/CMTB/waveModels/STWAVE/CBT2/Local-Field/Local-Field.ncml')

xshoreSpot = -1 #np.argmin(np.abs(HPnc['xFRF'][:] - 600))
yshoreSpot = np.argmin(np.abs(HPnc['yFRF'][:] - 945))

CBoffshoreHs = CBncfile['waveHs'][cb.astype(int), yshoreSpot, xshoreSpot]
HPoffshoreHs = HPnc['waveHs'][hp.astype(int), yshoreSpot, xshoreSpot]


### plotting all data for each individually
plt.figure()
plt.plot(nc.num2date(HPnc['time'][:], 'seconds since 1970-01-01'), HPnc['waveHs'][:, yshoreSpot, xshoreSpot], '.', label = 'HP')
plt.plot(nc.num2date(CBncfile['time'][:], 'seconds since 1970-01-01'), CBncfile['waveHs'][:, yshoreSpot, xshoreSpot], '.', label = 'CB')
plt.plot(nc.num2date(HPncStatic['time'][:], 'seconds since 1970-01-01'), HPncStatic['waveHs'][:, yshoreSpot, xshoreSpot], '.', label='Static')
plt.plot(nc.num2date(CBTncfile['time'][:], 'seconds since 1970-01-01'), CBTncfile['waveHs'][:, yshoreSpot, xshoreSpot], '.', label='Thresh')
plt.legend()





t, _, _ = sb.timeMatch(HPnc['time'][:], HPnc['waveHs'][:, yshoreSpot, xshoreSpot], CBncfile['time'][:], CBncfile['waveHs'][:, yshoreSpot, xshoreSpot])
t, _, _ = sb.timeMatch(t, range(len(t)), HPncStatic['time'][:], range(len(HPncStatic['time'][:])))
t, _, CBThs = sb.timeMatch(t, range(len(t)), CBTncfile['time'][:], CBTncfile['waveHs'][:, yshoreSpot, xshoreSpot])
_, _, HPShs = sb.timeMatch(t, range(len(t)), HPncStatic['time'][:], HPncStatic['waveHs'][:, yshoreSpot, xshoreSpot])
_, _, HPhs = sb.timeMatch(t, range(len(t)), HPnc['time'][:], HPnc['waveHs'][:, yshoreSpot, xshoreSpot])
_, _, CBhs = sb.timeMatch(t, range(len(t)), CBncfile['time'][:], CBncfile['waveHs'][:, yshoreSpot, xshoreSpot])



# look at 11 m awac Hsfor each
t, hphs, cbhs = sb.timeMatch(HPnc['time'][:], HPnc['waveHs'][:, yshoreSpot, xshoreSpot],  CBncfile['time'][:], CBncfile['waveHs'][:, yshoreSpot, xshoreSpot])
assert hphs - cbhs == 0
t, hphs, cbhs = sb.timeMatch(HPncStatic['time'][:], HPncStatic['waveHs'][:, yshoreSpot, xshoreSpot],  CBncfile['time'][:], CBncfile['waveHs'][:, yshoreSpot, xshoreSpot])
t, hphs, cbhs = sb.timeMatch(CBTncfile['time'][:], CBTncfile['waveHs'][:, yshoreSpot, xshoreSpot],  CBncfile['time'][:], CBncfile['waveHs'][:, yshoreSpot, xshoreSpot])

