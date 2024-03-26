import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pdb

cwd = os.getcwd()
dNames = os.listdir('AlluvialFan')
L = 30
N = 6
fig, axs = plt.subplots(L,N)
fig2, ax2 = plt.subplots()
colors = ['r', 'b', 'k', 'c', 'y', 'salmon']
#print(fNames)
j=0
for d in sorted(dNames):
    path = cwd + '/AlluvialFan/' + d
    fNames = os.listdir(path)
    k = 0
    zVar = []
    for f in sorted(fNames):
        print(f)
        fName = path + '/' + f
        df = pd.read_csv(fName, delimiter = '\t')
        #pdb.set_trace()
        x = df.iloc[:,0]
        lat = df.iloc[:,1]
        lon = df.iloc[:,2]
        z = df.iloc[:,3]
        
        x = np.array(x.to_list())
        lat = np.array(lat.to_list())
        lon = np.array(lon.to_list())
        z = np.array(z.to_list())
        ind = int(len(x)/2)
        middle = np.array([lat[ind], lon[ind]])
        axs[k,j].plot(x,z-np.mean(z), '-k')
        axs[k,j].set_ylim((-5, 5))
        axs[k,j].set_xlim((0, 2500))
        #axs[k].set_aspect('equal')
        if k==0:
            start = middle
            dist = 0
        else:
            distTemp = np.sqrt((middle[0]-start[0])**2 + (middle[1] - start[1])**2)
            dist = np.append(dist, distTemp)
            print(distTemp)
        zVar = np.append(zVar, np.var(z))
        k+=1

    ax2.plot(dist, zVar, 'o', color = colors[j])
    ax2.set_ylabel('Topographic Variance [m$^2$]')
    ax2.set_xlabel('Downfan Distance [m]')
    j+=1    
plt.show(block=False)
pdb.set_trace()