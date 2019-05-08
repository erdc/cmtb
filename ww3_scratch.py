from prepdata import inputOutput

ww3io = inputOutput.ww3IO('')
fname = '/home/spike/cmtb/grids/ww3/Mesh_interp_Mar2018.msh'
import matplotlib.pyplot as plt

data = ww3io.load_msh(fname)
plt.tripcolor(data['lon'], data['lat'], data['elevations'])
plt.colorbar()
plt.show()
