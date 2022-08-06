import numpy as np
from  my_lib_03 import *
from keras.models import load_model
import gc
"""
This file produces the dataset used in Pasquet et al. given the raw dataset.
(Needs a lot of available RAM, double the size of the dataset)
"""

file_path = "/sps/lsst/users/atheodor/sdss.npz" 
file = np.load(file_path)
images = file["cube"]
labels = file["labels"]
z = labels["z"]
der = labels["dered_petro_r"]
pr = labels["primtarget"]
dec = labels["dec"]
ra = labels["ra"]


i=0
pasq = []
z1 = []
der1 = []
dec1 = []
ra1 = []

for a in images:
    if ((z[i] <= 0.4) and (der[i] <=17.8) and (pr[i]>=64)) :
        z1.append(z[i])
        der1.append(der[i])
        dec1.append(dec[i])
        ra1.append(ra[i])
        i += 1
    else:
        i +=1

#Dict = {'z': z1, 'der_petr_r': der1, 'dec': dec1,'ra':ra1}
# Pasquet dataset (images, redshift)
np.savez("/sps/lsst/users/atheodor/pasq_01.npz", images = images,label = z1)

#np.savez("/sps/lsst/users/atheodor/pasq_03.npz", images = z1,label = der1)
#np.savez("/sps/lsst/users/atheodor/pasq_04.npz", images = dec1,label = ra1)





