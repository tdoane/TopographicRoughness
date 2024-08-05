import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.sparse import coo_matrix, lil_matrix, identity, diags
#from statsmodels.tsa.stattools import acf
import pdb

def InverseSample(noise, mode):
    ## Take a Normally distributed noise and turn it into a noise with a different underlying distribution
    noise/=(np.sqrt(2)*np.std(noise))
    noise-=np.mean(noise)
    Fval = (1+erf(noise))/2.
    
    if mode=='exp':
    ## exponential distribution
        lam = 3.
        samples = -np.log(Fval)*lam
        samples[np.isinf(samples)]=0. 
    if mode=='Lomax':
    ##Lomax Distribution
        lam = 1.
        beta = 5.
        samples = lam*((1-Fval)**(-1/beta)-1)
        samples[np.isinf(samples)]=0. 
    
    if mode=='Weibull':
    ##Weibull Distribution
        lam = .1
        beta = 0.5
        samples = lam*((-np.log(Fval))**(1/beta))

    #samples/=10000

    return(samples)
def ARmake(L, phi):
    z = np.zeros(int(L))
    x = np.arange(0, len(z))
    for i in np.arange(1,int(L)):
        z[i] = phi*z[i-1] + np.random.normal(0, 0.1)
    fit = np.polyfit(x,z,1)
    z-=np.polyval(fit,x)
    return(z)
def makeFeature(mode, XX, YY):
    if mode=='mound':
        feat = A*np.exp(-XX**2/L**2 - YY**2/W**2)
    elif mode =='pit-mound':
        feat = -2*A*XX/L**2*np.exp(-XX**2/L**2 - YY**2/W**2)
    elif mode == 'crater':
        A0 = 0.126*L
        L0 = 0.85*L
        L2 = A0*L**2/2.
        A2 = L2**2*A0/8.
        r = np.sqrt(XX**2 + YY**2)
        feat = A2*np.exp(-r**2/(L2**2))*(-2./(L2**2) + 4.*r**2/(L2**4)) - A0*np.exp(-r**2/(L0**2))
        #pdb.set_trace()
    return(feat)
def matMake(z, D, dx, dt):
    lY, lX = np.shape(z)# size of the domain

    #dt = dx**2/(8*D)
    #dt=1. #Define a time-step
    
    mVal = -4*D*dt/(dx**2)+1 #define middle values of the stencil central difference solution
    oVal = D*dt/(dx**2) #define adjacent values for the stencil of the central difference solution
    
    lInd = np.arange(lY-1, lY*lX-1, lY)
    rInd = np.arange(lY-1, lY*lX-1, lY)
    
    ind2 = np.arange(0, lY*lX-lY+1, lY)
    BCVec = np.zeros(lY*lX-lY+1)
    BCVec[ind2] = oVal
    
    lVec = np.ones(lX*lY-1)*oVal #make a vector of values to construct the sparse matrix
    lVec2 = np.ones(lY*lX-lY)*oVal
    lVec[lInd]=0 #Boundary Conditions
    lVecBC = np.zeros(lX*lY-(lY-1))
    
    rVec = np.ones(lY*lX-1)*oVal #make a vecor of values to construct the sparce matrix
    rVec2 = np.ones(lY*lX-lY)*oVal
    rVec[rInd]=0 #Boundary Conditions
    mVec = np.ones(lX*lY)*mVal #make a vector of values to construct the sparse matrix
    
    mat0=identity(lY*lX, dtype='float')*mVal
    
    matL1 = diags(lVec, -1, dtype='float', shape = (lX*lY, lX*lY))
    matU1 = diags(rVec, 1, dtype = 'float', shape = (lX*lY, lX*lY))
    
    matL2 = diags(lVec2, -lX, dtype ='float', shape = (lX*lY, lX*lY))
    matU2 = diags(rVec2, lX, dtype = 'float', shape=(lX*lY, lX*lY))
    
    matBC1 = diags(BCVec, -lX+1, dtype = 'float', shape=(lX*lY, lX*lY))
    matBC2 = diags(BCVec, lX-1, dtype = 'float', shape=(lX*lY, lX*lY))
   
    #boundary conditions: perdiodic boundary conditions
    matU3 = diags(np.ones(lX)*oVal, lX*lY-(lX), dtype = 'float', shape=(lX*lY, lX*lY))
    matL3 = diags(np.ones(lX)*oVal, -lX*lY+(lX), dtype='float', shape=(lY*lY, lX*lY))
    
    mat = mat0 + matL1 + matL2 + matU1 + matU2 + matU3 + matL3 + matBC1 + matBC2

    return(mat)

def hillshade(array,azimuth,angle_altitude):
    #Create hillshade from topography
    azimuth = 360.0 - azimuth

    xx, yy = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(xx*xx + yy*yy))
    aspect = np.arctan2(-xx, yy)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.

    shaded = np.sin(altituderad)*np.sin(slope) + np.cos(altituderad)*np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)
    return(shaded)
    
dx = 0.1
xLim = 10
yLim = 10
x = np.arange(-xLim, xLim, dx)
y = np.arange(-yLim, yLim, dx)
X, Y = np.meshgrid(x, y)
lX, lY = len(x), len(y)

H = (2*xLim)**2

spatialDensity = 550/10000
numShrubs = int(spatialDensity*(2*xLim)**2)
xPos = np.random.randint(0, lX, size = numShrubs)
yPos = np.random.randint(0, lY, size = numShrubs)
#pdb.set_trace()

xPos = np.random.uniform(-xLim, xLim,size=numShrubs)
yPos = np.random.uniform(-yLim, yLim, size=numShrubs)

A = 0.2
L = 0.25
W = 0.25
K = 0.0025
#dt = int(dx**2/(8*K))
dt=1
N = 10000
T = int(N/dt)

numReplace = np.array([4,8,12])*(H/10000)

mat = matMake(X, K, dx, dt)

accent = plt.get_cmap('RdBu')

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots(3,1)

SS = np.pi*A**2*L**2/(2*(2*xLim)**2)
zMu = np.pi*A*L**2/((2*xLim)**2)
j=0
for p in numReplace:
    count=0
    while count<1:
        t=0
        varZ = np.zeros(T)
        muZ = np.zeros(T)
        z = np.zeros_like(X)
        tTemp = 0
        k=0
        clr = accent(float(p/0.5))
        zShrub = np.zeros_like(z)
        for i in range(len(xPos)):
            x0, y0 = xPos[i], yPos[i]
            XTemp, YTemp = X-x0, Y-y0
            XTemp[XTemp>xLim]-=2*xLim
            XTemp[XTemp<-xLim]+=2*xLim
            YTemp[YTemp>yLim]-=2*yLim
            YTemp[YTemp<-yLim]+=2*yLim
            feat = makeFeature('mound', XTemp, YTemp)
            zShrub+=feat
        while t<T:
            switch = np.random.binomial(1, p)
            if switch==1:
                iTemp = np.random.randint(0, numShrubs,size=1)
                for ind in iTemp:
                    indX = xPos[ind]
                    indY = yPos[ind]
                    
                    #x0, y0 = x[indX], y[indY]
                    x0, y0 = xPos[ind], yPos[ind]
                    XTemp, YTemp = X-x0, Y-y0
                    XTemp[XTemp>xLim]-=2*xLim
                    XTemp[XTemp<-xLim]+=2*xLim
                    YTemp[YTemp>yLim]-=2*yLim
                    YTemp[YTemp<-yLim]+=2*yLim
                    feat = makeFeature('mound', XTemp, YTemp)
                    z+=feat
                    zShrub-=feat
                    
                    xPos[ind]=np.random.uniform(-xLim, xLim, size=1)
                    yPos[ind]=np.random.uniform(-yLim, yLim, size=1)

                    x0, y0 = xPos[ind], yPos[ind]
                    XTemp, YTemp = X-x0, Y-y0
                    XTemp[XTemp>xLim]-=2*xLim
                    XTemp[XTemp<-xLim]+=2*xLim
                    YTemp[YTemp>yLim]-=2*yLim
                    YTemp[YTemp<-yLim]+=2*yLim
                    feat = makeFeature('mound', XTemp, YTemp)
                    zShrub+=feat
                    #xPos[ind]=np.random.randint(0, lX)
                    #yPos[ind]=np.random.randint(0, lY)
                

                    
                    #if np.remainder(t,10)==0:
                # if np.remainder(t,10)==0:
                #     zShrub=np.zeros_like(z)        
                #     for i in range(numShrubs):
                        
                #         indX = xPos[i]
                #         indY = yPos[i]
                #         #xNew[ind]=np.random.randint(0, lX)
                #         #yNew[ind]=np.random.randint(0, lY)
                #         x0, y0 = x[indX], y[indY]
                #         XTemp, YTemp = X-x0, Y-y0
                #         XTemp[XTemp>xLim]-=2*xLim
                #         XTemp[XTemp<-xLim]+=2*xLim
                #         YTemp[YTemp>yLim]-=2*yLim
                #         YTemp[YTemp<-yLim]+=2*yLim
                #         feat = makeFeature('mound', XTemp, YTemp)
                #         zShrub+=feat

            z=mat.dot(z.ravel(order='F'))
            z = np.reshape(z, (lX, lY), order = 'F',)

            varZ[t] = np.var(z+zShrub)

            muZ[t] = np.mean(z)

            #else:
            #    varZ[t] = np.var(z) + numShrubs*(SS - zMu**2)
            #    muZ[t] = np.mean(z) + numShrubs*zMu

            if np.isnan(np.var(z.ravel(order = 'F'))):
                print('unstable numerical simulation')
                break
            t+=1
            if np.remainder(t,100)==0:
                print(t)

        #muVar = -np.mean(p)/((2*xLim)**2)*np.pi*A**2*L**4*np.log(L**2)/(8*K)
        
        T0 = H/(8*K*np.pi) - L**2/(4*K)

        t2 = np.arange(0, T0)
        SS2 = np.pi*A**2*L**4/(2*(2*xLim)**2*(L**2 + 4*K*t2))
        zMu2 = np.pi*A*L**2/((2*xLim)**2)#*np.mean(p)*t2
        varTemp = SS2-zMu**2
        muVar = np.trapz(np.mean(p)*varTemp, t2) + numShrubs*(SS + zMu**2)
        #zMu = zMu*np.mean(p)*t2

        Sa = numShrubs#175.*(H/10000)
        phi = p/(1*Sa)
        Theory = Sa*A**2*L**2*np.pi/(2*H)*(1 + phi*L**2/(4*K)*(np.log(1 + 4*K*T0/L**2) - 8*np.pi*K*T0/H))
        #Theory = Sa*A**2*L**2*np.pi/(2*H)*(1 + p/Sa*L**2/(4*K)*(np.log(1 + 4*K*T0/L**2) - 8*np.pi*K*T0/H))
        Theory = Sa*A**2*L**2*np.pi/(2*H)*(1 - (2*np.pi*L**2)/H) + phi*Sa*A**2*L**4*np.pi/(8*K*H)*(np.log(1 + 4*K*T0/L**2) - 8*np.pi*K*T0/H)

        hShade = hillshade(100*(z+zShrub), 270, 5)
        grad = np.gradient(z + zShrub, dx)
        slope = np.sqrt(grad[0]**2 + grad[1]**2)
        #corr = acf(p, 100)

        ax1.plot(varZ, '-', color = clr, alpha = 0.5)
        #ax1.plot([0, t], [muVar, muVar], '-k')
        
        #ax1.plot(p*np.cumsum(varTemp)+numShrubs*(SS + zMu**2), '-k')
        
    
        #im=ax2[j].imshow(slope, cmap = 'viridis')
        count+=1
    Sa = 250.*(H/10000)
    phi = p/(1*numShrubs)
    #Theory = numShrubs*A**2*L**2*np.pi/(2*H)*(1 + phi*L**2/(4*K)*(np.log(1 + 4*K*T0/L**2) - 8*np.pi*K*T0/H))
    Theory = Sa*A**2*L**2*np.pi/(2*H)*(1 - (2*np.pi*L**2)/H) + phi*Sa*A**2*L**4*np.pi/(8*K*H)*(np.log(1 + 4*K*T0/L**2) - 8*np.pi*K*T0/H)

    ax2[j].imshow(hShade, cmap = 'gray')
    ax1.plot([0,t], [Theory, Theory], '-', color=clr)
    ax1.plot([0, t], [Theory, Theory], '--k')
    j+=1

#cbar_ax = fig2.add_axes([0.925, 0.15, 0.025, 0.7])
#colbar=fig2.colorbar(im, cax=cbar_ax, )    

plt.show(block = False)
pdb.set_trace()