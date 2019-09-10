import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
font = {'family' : 'normal',
    'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import pandas as pd
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from fbm import FBM,MBM
import numpy as np
import random

nlength = 14
#Load model
model = keras.models.load_model(".\\good_DLFNNmodels\\model3densediff_n"+str(nlength-1)+".h5")
#generate fbm example
simulatedH = []
testH = []
for samples in range(0,1000):
    Hsim = random.uniform(0,1)
    f = FBM(n=nlength-1,hurst=Hsim,length=1,method='hosking')
    x = np.transpose(pd.DataFrame(f.fbm()).values)
    xdiff = np.transpose(pd.DataFrame(f.fbm()).values[1:])
    for j in range(1,len(x)):
        xdiff[0,j-1] = x[j]-x[j-1]
        # /(np.amax(x)-np.amin(x))
    # print(x)
    # print(xdiff)
    exp_est = model.predict(xdiff)
    simulatedH.append(Hsim)
    testH.append(exp_est[0][0])
    print('simulated: ',Hsim,'predicted: ',exp_est)

plt.figure()
plt.plot(simulatedH,testH,'b.')
plt.plot([0,1],[0,1],'r-')
plt.xlabel('H simulated')
plt.ylabel('H estimated')
plt.show()
