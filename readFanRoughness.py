import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import os
from statsmodels.nonparametric.smoothers_lowess import lowess 
import pdb
from scipy.optimize import curve_fit

def matMake1D(z, dx, K):
    lX = len(z)
    mVal = -2.*K*dt/dx**2 + 1.
    oVal = dt*K/dx**2
    
    mVec = np.ones(lX)*mVal
    mVec[0]=1.
    mVec[-1]=1.
    vec = np.ones(lX-1)*oVal
    vec[0]*=0.
    
    mat0 = diags(mVec, 0, shape = (lX, lX))
    mat1 = diags(vec, 1, shape = (lX, lX))
    mat2=diags(np.flip(vec), -1, shape=(lX, lX))
    mat = mat0 + mat1 + mat2
    return(mat)

def kernelDensity(x, z, L):
    h=0
    zCollect = np.zeros_like(x)
    for i in x:
        weight = 1/np.sqrt(2*np.pi*L**2)*np.exp(-(x-i)**2/(2*L**2))
        weight /= np.sum(weight)
        zNew  = np.sum(z*weight)
        zCollect[h]=zNew
        h+=1
    return(zCollect)
        
def smoothContours(cs):
    xCollect = {}
    yCollect = {}
    #pdb.set_trace()
    temp = cs.allsegs
    N = len(temp)
    #db.set_trace()
    for k in np.arange(N-1,0, -1):
        temp2 = temp[k][:][0]
        kStr = str(k)
        xCollect[kStr]=[]
        yCollect[kStr]=[]
        
        #pdb.set_trace()
    
        temp3 = np.array(temp2).squeeze()
        try:
            x = temp3[:,0]
            y = temp3[:,1]
        except:
            pdb.set_trace()
        dx = np.sqrt((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2)
        s = np.cumsum(dx)
        s = np.append(0, s)
        yNew = kernelDensity(s,y,100)
        xNew = kernelDensity(s,x,100)
            #cpdb.set_trace()
        if len(yNew)<40:
            print(k, 'continuing')
                #pdb.set_trace()
            continue
        xCollect[kStr].append(xNew)
        yCollect[kStr].append(yNew)
                
                #pdb.set_trace()
        #k+=1
    return(xCollect,yCollect)

def hillshade(array,azimuth,angle_altitude):
    azimuth = 360.0 - azimuth

    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.

    shaded = np.sin(altituderad)*np.sin(slope) + np.cos(altituderad)*np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)

    return(255*(shaded + 1)/2)

def invertFun(x, A, B):
    return A*x**(-1) + B

def widthDistFun(x, y, A, B, C):
    return B - A*(x/y)

fold = 'Data/'
fNames = os.listdir(fold)

fig0, ax0 = plt.subplots()

clrs = plt.get_cmap('tab20')
C=0
fNames=['Fan_Example.tif']
for f in fNames:
    clr = clrs(float(C/20)) 
    pName = fold + f
    raster = gdal.Open(pName)
    xMin, dx, _, yMax, _, dy = raster.GetGeoTransform()

    z = raster.ReadAsArray()
    z[z<=0]=np.nan
    
    lY, lX = np.shape(z)

    xTickLabs=np.arange(int(xMin), int(xMin)+lX, 1000).astype('str')
    yTickLabs=np.arange(int(yMax), int(yMax)-lY, -1000).astype('str')

    xTix=np.arange(0, lX, 1000)
    yTix=np.arange(0, lY, 1000)

    hShade = hillshade(z, 315, 15)

    lam = 50.
    elevs = np.arange(np.round(np.nanmin(z.ravel())+2), np.floor(np.nanmax(z.ravel())-1), 5.)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(hShade, cmap = 'gray')
    cs = ax[0].contour(z, elevs)
    ax[0].set_xticks(xTix)
    ax[0].set_yticks(yTix)
    ax[0].set_yticklabels(yTickLabs)
    ax[0].set_xticklabels(xTickLabs)
    ax[0].set_title('Smoothed Contours')

    temp=cs.allsegs
    N=len(temp)

    xNew={}
    yNew={}

    fig1, ax1 =plt.subplots(2,1)
    for k in np.arange(N-1,0,-1):
        temp2=temp[k][:][0]
        kStr=str(k)

        if k==3 or k==20:
            temp2=temp[k][:][1]

        temp3=np.array(temp2).squeeze()
        x=temp3[:,0]
        y=temp3[:,1]

        dx = np.sqrt((x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2)
        s = np.cumsum(dx)
        s = np.append(0, s)
        ySmooth = kernelDensity(s,y,100)
        xSmooth = kernelDensity(s,x,100)

        if len(s)<40:
            print('Contour too short, skipping')
            continue

        xNew[kStr]=[]
        yNew[kStr]=[]

        xNew[kStr].append(xSmooth)
        yNew[kStr].append(ySmooth)
        
    lX, lY = np.shape(z)
    kk = 0
    j=0
    varCollect = []
    elevs = []
    lenCollect =[]

    try: 
        xNew.keys()
        print('keys exist')
    except:
        pdb.set_trace()

    for key in xNew.keys():
        try:
            xTemp = xNew[key][0]
            yTemp = yNew[key][0]
        except:
            print('Error')
            pdb.set_trace()
            continue
        #print(len(xTemp))

        dx = np.sqrt((xTemp[1:]-xTemp[:-1])**2 + (yTemp[1:]-yTemp[:-1])**2)
        s = np.cumsum(dx)
        s = np.append(0, s)
        s2=np.arange(0, s[-1], 1.)
        x = np.interp(s2, s, xTemp)
        y = np.interp(s2, s, yTemp)

        if (kk==4 or kk==20):
            ax[0].plot(x,y, '-r')
        else:
            ax[0].plot(x, y, '-', color = clr)

        xInd = np.round(x)
        yInd = np.round(y)
        vect = yInd + xInd*lY
        dup = [idx for idx, item in enumerate(vect) if item in vect[:idx]]
        #print(dup)
        xInd = np.delete(xInd, dup)
        yInd = np.delete(yInd, dup)
        dx = np.sqrt((xInd[1:]-xInd[:-1])**2 + (yInd[1:]-yInd[:-1])**2)
        yInd = [int(indY) for indY in yInd]
        xInd = [int(indX) for indX in xInd]
        zTemp = z[yInd, xInd]


        midInd = int(len(xInd)/2)
        if kk==0:
            x0 = xInd[midInd]
            y0 = yInd[midInd]
            dist = 0.
            elevs = np.mean(zTemp)
        else:
            xM = xInd[midInd]
            yM = yInd[midInd]
            dist = np.append(dist, np.sqrt((xM - x0)**2 + (yM - y0)**2))
            elevs = np.append(elevs, np.mean(zTemp))
        
        s = np.append(0, np.cumsum(dx))

        varCollect = np.append(varCollect, np.var(zTemp))
        lenCollect =np.append(lenCollect, len(zTemp))

        if (kk==4 or kk==20):
            ax1[j].plot(s, zTemp-np.mean(zTemp), '-r')
            ax1[j].set_ylim(-1, 1)
            ax1[j].set_xlim(0, 2150)
            j+=1

        kk+=1

    cutoff=0

    try:
        dist_p, dist_cov = curve_fit(invertFun, dist[cutoff:], varCollect[cutoff:])
        A, B = dist_p
        ax[1].plot(dist[cutoff:], A/dist[cutoff:] +B, '.k', label='r=a/distance + c')
    except:
        print('did not work')

    try:
        width_p, width_cov = curve_fit(invertFun, lenCollect[cutoff:], varCollect[cutoff:])
        A, B=width_p
        ax[1].plot(dist[cutoff:], A/lenCollect[cutoff:]+B, '-k', label='r=a/width + c')
    except:
        print('did not work')

    ax
    ax[1].plot(dist[cutoff:], varCollect[cutoff:], 'o', color = clr, label='data')
    #ax[2].plot(dist[cutoff:], varCollect[cutoff:], 'o', color = clr)
    #ax[3].plot(dist[cutoff:], varCollect[cutoff:], 'o', color = clr)

    ax[1].set_title('roughness vs. downfan distance')
    ax[1].legend()
    #ax[3].set_title('Width/Distance')
    ax[0].set_ylabel('Northing')
    ax[0].set_xlabel('Easting')

    ax0.plot(dist, varCollect, 'o', color = clr)
    ax[0].set_aspect('equal')
    slope = np.tan((elevs[0] - elevs[-1])/(dist[-1]))*180./np.pi
   # pdb.set_trace()
    #fig.text(0.95, 0.95, 'slope =' + str(slope),
    #    verticalalignment='bottom', horizontalalignment='right',
    #    color='black', fontsize=15)
    C+=1
    #fig.savefig('\home\\tyler\Research\RoughnessGeneral\Figures\\' + str(C))
    #ax0.set_xlabel('Downfan Distance [m]')
    #ax0.set_ylabel('Topographic Variance')
    #plt.figure()
    #plt.plot(pos, theta)
    #plt.plot(temp[:,0], temp[:,1])
    #plt.plot(xNew, yNew)
plt.show()
