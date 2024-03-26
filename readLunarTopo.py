from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pdb

def lowPass(z, l0):
    #create a low-pass filter that smooths topography using a Gaussian kernel
    from scipy.signal import detrend
    lY, lX = np.shape(z)
    x, y = np.arange(-lX/2, lX/2), np.arange(-lY/2, lY/2)
    X, Y = np.meshgrid(x, y)
    filt = 1/(2*np.pi*l0**2)*np.exp(-(X**2 + Y**2)/(2*l0**2))
    ftFilt = np.fft.fft2(filt)
    #z = detrend(z, axis=0)
    #z = detrend(z, axis=-1)
    ftZ = np.fft.fft2(z)
    ftZNew = ftZ*ftFilt
    zNew = np.fft.ifft2(ftZNew).real
    zNew = np.fft.fftshift(zNew)
    return(zNew)

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]
def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

myColors = np.load('myColors.npy', allow_pickle = 'True').item()
hexlist = myColors['Arizona Sunset']

clrs = get_continuous_cmap(hexlist)

fig, axs=plt.subplots(nrows=2,ncols=3)
fig2, ax2=plt.subplots()
fig3, ax3=plt.subplots()
fig4, ax4=plt.subplots()

axs=axs.flatten()

fName = 'Data/MoonHigh.tif' #Navigate to GeoTiff
raster = gdal.Open(fName) #open GeoTiff
z = raster.ReadAsArray() #Import as numpy array

dx = int(np.round(raster.GetGeoTransform()[1])) #Get Raster Resolution
 
L = np.array([5., 12.5, 25., 50., 100., 200])/dx #Create Vector of Lengthscales for smoothing
hPassOld = np.zeros_like(z) #Initialize a vector that forms the lower limit for the band of length scales
varCollect=[] #initialize a vector to collect variance values
k=0 #A Counter
lX, lY = np.shape(z)
x = np.arange(0,lX*dx, 0.1)

gaussOld=np.zeros_like(x) #Initialize a function that will form one lower limit for the band-pass topography
colors=['black', 'firebrick', 'orange', 'olivedrab', 'mediumturquoise', 'dodgerblue']
for i, ax in enumerate(axs):#Loop through smoothing length scales
    L0=L[i]
    lPass = lowPass(z, L0) #Create low pass filter
    hPass = z - lPass #Create High-pass filter
    band=hPass-hPassOld #Create Band-pass topography
    #hPassOld=hPass
    varCollect=np.append(varCollect, np.var(band[3*int(L0):-3*int(L0), 3*int(L0):-3*int(L0)])) #Calculate roughness over areas not affected by edge effects
    im=ax.imshow(band[3*int(L0):-3*int(L0), 3*int(L0):-3*int(L0)], vmin=-L0, vmax=L0, cmap = clrs) #Plot band-pass topography
    tcks=np.arange(0, np.shape(band)[0]-6*int(L0), 500)
    ax.set_yticks(tcks)
    ax.set_xticks(tcks)
    gauss = np.exp(-x**2/L0**2) #create function for plotting
    col=colors[k] 
    ax2.fill_between(x, gaussOld, gauss, color=col) #Plot band-pass scales
    ax3.loglog(L0, varCollect[k], 'o', color=col) 
    ax4.plot(band[1000,3*int(L[-1]):-3*int(L[-1])], '-', color = col)
    gaussOld=gauss #Replace Upper limit band-pass becomes the lower limit for the next iteration
    k+=1
    print(k)

ax3.grid()
#ax2.set_yscale('log')
ax2.set_xscale('log')

cbar_ax = fig.add_axes([0.925, 0.15, 0.025, 0.7])
colbar=fig.colorbar(im, cax=cbar_ax, )
colbar.set_ticks([-L0, L0])

plt.show(block=False)
pdb.set_trace()