import numpy as np
import tensorflow as tf
from  Photo_z_lib import *
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import gc
from sklearn.model_selection import train_test_split

Nbins = 150
red_max = 0.4
images_train = []
labels_train = []
images, labels = load_data("datal_001.npz")

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)

'''
images1, labels1 = load_data("data_01.npz")
images2, labels2 = load_data("data_02.npz")
images3, labels3 = load_data("data_03.npz")
images4, labels4 = load_data("data_04.npz")
images = np.concatenate((images1,images2,images3,images4),axis = 0)
labels = np.concatenate((labels1,labels2,labels3,labels4),axis = 0)

del images1
del images2
del labels1
del labels2
del images3
del labels3
del images4
del labels4
gc.collect()

i = 0
for a in images1:
  if labels1[i]<red_max:
    images_train.append(a)
    labels_train.append(labels1[i])
    i+=1
  else: 
    i+=1
i=0


del images1
del labels1
gc.collect()
'''
#test_images, test_labels = load_data("data_05.npz")

run_model_with_data('Pasquet',(64,64,5),X_train, y_train, X_test,y_test,Nbins,8,name = 'model_002',Augmentation = True,Pooling = True, Dropout = False,batch_size = 64,epochs = 30,validation_split = 0.2,monitor = 'loss')
