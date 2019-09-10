import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
font = {'family' : 'normal',
    'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

import pandas as pd
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from fbm import FBM,MBM
import numpy as np
import random

nlength = 14
inclength = 40
#Load model
model = keras.models.load_model(".\\good_DLFNNmodels\\model3densediff_n"+str(nlength-1)+".h5")
multipath = []
multiexp = []
estmultiexp = []
esttime = []

#stitch together multifractional fbm
for i in range(0,10):
    Hsim = random.uniform(0,1)
    randinclength = random.randrange(20)+inclength
    f = FBM(n=randinclength,hurst=Hsim,length=1,method='hosking')
    x = f.fbm()
    if i == 0:
        for p in x:
            multipath.append(p)
            multiexp.append(Hsim)
    else:
        checkpoint = multipath[-1]
        for p in x[1:]:
            multipath.append(checkpoint+p)
            multiexp.append(Hsim)

#symmetric window analysis
eitherside = int(np.floor(float(nlength)/2.))
for i in range(eitherside,len(multipath)-eitherside):
    #for differencing
    if nlength % 2 == 1:
        x = np.transpose(pd.DataFrame(multipath[i-eitherside:i+eitherside+1]-multipath[i-eitherside]).values)
    else:
        x = np.transpose(pd.DataFrame(multipath[i-eitherside:i+eitherside]-multipath[i-eitherside]).values)
    xshape = x.shape
    xdiff = np.empty((xshape[0],xshape[1]-1))
    for j in range(1,len(x[0])):
        xdiff[0][j-1] = (x[0][j]-x[0][j-1])
        # /(np.amax(x[0])-np.amin(x[0]))
    test_one = model.predict(xdiff)
    esttime.append(i)
    estmultiexp.append(0.5*test_one[0][0])

#plots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(multipath,'-',lw=1)
ax1.set_ylabel('x (A.U.)')
ax1.set_title('Simulated data')

ax2.plot(multiexp,'m-',lw=3,label='simulated',alpha=0.5)
ax2.plot(esttime,estmultiexp,'k-',lw=1,label='predicted')
ax2.plot([0.,len(multipath)],[1.,1.],'r-',lw=0.5)
ax2.legend(fontsize=10,loc='upper center', bbox_to_anchor=(0.5, 1.22),ncol=2,fancybox=True,shadow=True)
ax2.set_xlabel('Time points')
ax2.set_ylabel(r'$H$')
ax2.set_xlim(0.,len(multipath))
ax2.set_ylim(0.,1.)

plt.tight_layout()
plt.show()
