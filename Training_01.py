#Script training 4 models with different # layers # amount of data and saving them 
import numpy as np
import tensorflow as tf
from  Photo_z_lib import *
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
#from keras.models import load_model
import gc

# Choosing the number of Bins and Layers
Nbins = 180
Nlayers = 5

# Choose the model type
model_type = 'seq'
#model_type = 'Pasquet'

# Loading the data
file_path = "/sps/lsst/users/atheodor/pasq_02.npz"
file = np.load(file_path)
images = file["images"]
labels = file["label"]
images = np.asarray(images)

# Splitting the data into training and test data
images1, test_images, labels1, test_labels = train_test_split(images, labels,train_size = 300000, test_size=50000, random_state=42)
#print(len(labels))
del images
gc.collect()

# Used to produce a name easy to distinguish during prototyping.

train_d = int(len(images1)/10000)
test_d = int(len(test_images)/10000)
model_name = model_type + '_' + str(Nbins) + '_' + str(Nlayers) + '_' + str(train_d) + '_' + str(test_d)
plot_name = 'Plot_' + model_name
res_name = 'Residuals' + model_name

# Compiles and trains the model
model = run_model_with_data(model_type,(64,64,5),images1,labels1,test_images,test_labels,Nbins,Nlayers,name  = model_name,Pooling = False,epochs = 30,batch_size = 32)

