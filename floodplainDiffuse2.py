import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import identity, diags
from scipy.signal import find_peaks
import pdb

##This script simulates the evolution of a fan surface that has random avulsions that leave abandoned channel forms that then diffuse by hillslope processes. 

#plt.rc('text', usetex = 'True')
#plt.rc('font', size = 12)
#plt.rc('font', serif='True')
#plt.rc('font', family = 'Times')


A0 = 2. #Amplitude of Gaussian
A2 = 10. #Amplitude of second derivative of Gaussian
L =2.5  #Length scale of Gaussian   
K = 0.05 #Topographic Diffusivity
dt = 1. #Time step

lX = 500 #Length of contour
dx = 1. #spatial step length
x = np.arange(-lX/2*dx, lX/2*dx, dx) #Array of positions along contour
 
H = lX*dx #physical length of domain
T0 = L**2/(4*K)*(H**2/(2*np.pi*L**2)-1) #Define a saturation timescale

def mkChan(x, mode): #Function fo make a channel with various combinations of zero and second order Gaussians
    if mode =='levee':
        chan = -A0*np.exp(-x**2/L**2) + (-2./(L**2) + 4.*x**2/(L**4))*A2*np.exp(-x**2/L**2)
    elif mode=='no levee':
        chan = -A0*np.exp(-x**2/L**2)
    return(chan)

def matMake(lX): #Create sparse matrices for diffusion
    mVal = -2*K*dt/(dx**2) + 1.
    oVal = K*dt/(dx**2)
    
    rVec = np.ones(lX-1)*oVal
    rVec[0]*=2.
    
    lVec = np.ones(lX-1)*oVal
    lVec[-1]*=2
    
    mat0 = identity(lX)*mVal
    mat1 = diags(rVec, 1, dtype = 'float', shape = (lX, lX))
    mat2 = diags(lVec, -1, dtype = 'float', shape = (lX, lX))
    mat = mat0 + mat1 + mat2
    return(mat)


T = 500000 #Duration of run

clrs = plt.get_cmap('RdBu') #Color maps
mat = matMake(lX) #Make matrices
prob = np.arange(0.0001, 0.01, 0.0005) #loop through probababilities of avulsion frequency
fig, axs = plt.subplots(3,1) #
fig2, axs2 = plt.subplots()
fig3, axs3 = plt.subplots()

window = 5*int(L)
k=0

for p in [0.001, 0.005, 0.025]: #Define probability of avulsions per unit time
    count=0
    while count<5:
        t = 0
        z = np.zeros(lX)
        zVar = np.zeros(T)
        zMu = np.zeros(T)
        #pdb.set_trace()
        switch = np.random.binomial(1, p, size = T) #Create vector of times when avulsions will happen
        while t<T:
            if switch[t]==1:
                ind = int((np.random.rand(1))*lX) #Randomly assign new channel location
                print(t/T)
                x0 = x[ind] #Identify physical location
                xTemp = x-x0  #Adjust temporary array to be centered on new array  
                zTemp = z[ind-window:ind+window] #take slice of topogrpahy around array
                newChan = mkChan(xTemp, 'no levee') #Make new channel and include the pre-existing topography
                chanTemp = newChan[ind-window:ind+window] #clip section of new channel
                chanTemp+=np.mean(zTemp) #adjust to the height of the clipped section
                newChan[ind-window:ind+window] =chanTemp-zTemp #add new channel
                z+=newChan #emplace in topography
                #plt.plot(z)
                #plt.show()
            
            z = mat.dot(z) #Diffuse topography
            zVar[t] = np.var(z) #Calculate statistics
            zMu[t] = np.mean(z)
            t+=1
        clr = clrs(float((k+1)/3))
        if count==4:
            axs[k].plot(x,z-np.mean(z), '-', color = clr)
            axs[k].set_ylim(-3, 3)
        muVar = np.mean(zVar[-500000:])
        LPoverHT0 = L*p/(H*T0)
        axs3.plot(LPoverHT0, muVar, 'ok')
        
        
        #Plotting routines
        muP = np.sum(switch)/T
        H = lX*dx
        T0 = L**2/(4*K)*(H**2/(2*np.pi*L**2)-1)
        t2 = np.arange(0, T0)
        SS =  np.sqrt(np.pi)*A0**2*L**2/(2*np.sqrt(2)*H)*(L**2/4 + K*t2)**(-1/2)

        muZ = np.pi**(1/2)*A0*L/H
        varTemp = SS - muZ**2
        #Theory for expected topographic roughness
        Theory = muP*(np.sqrt(np.pi)*A0**2*L**3/(2*np.sqrt(2)*K*H)*(H/(L*np.sqrt(2*np.pi)) -1) - np.sqrt(np.pi)*A0**2*L**4/(4*H**2*K)*(H**2/(2*np.pi*L**2)-1))

        axs2.plot(zVar, '-', color = clr, alpha = 0.5)
        count+=1
        print(k,count)
    axs2.plot([0, t], [Theory, Theory], '-k')
    axs2.plot([0,t], [Theory, Theory], '--', color=clr) 
    k+=1

plt.show(block=False)
pdb.set_trace()
