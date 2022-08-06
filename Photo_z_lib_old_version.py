import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import gc

''' Loads the data from an .npz file, returns the images as numpy arrays and the redshifts '''
def load_data(file_name):
    file_path = "/sps/lsst/users/atheodor/" + file_name
    file = np.load(file_path)
    images = file["images"]
    labels = file["label"]
    images = np.asarray(images)
    return images,labels 




''' Making the bins '''
def make_classes(red_max,Nbins,labels):
    #red_max = 1. # When the redshift is 1 maybe need to do Nbins + 1 in line 77 Use double values to work properly e.g. 1. not 1
    i = 0
    zbins = []
    redshifts = []
    #Nbins = 150

    for a in range(0,Nbins,1): # one bin is added by np.digitise 151 in total
        zbins.append(i/Nbins)
        i+=1
    i = 0
    zbins = np.asarray(zbins)*red_max


    # Prepare data for training 

    yyy = np.digitize(labels,zbins,right = True)
    return yyy


''' Defining the inception module from the Pasquet paper'''

def pasq_inception_module(x,f1,f2,f3):
  #1x1 convolution
  conv11 = tf.keras.layers.Conv2D(f1,(1,1),padding = 'same',activation = 'relu')(x)
  
  #3x3 convolution
  conv13 = tf.keras.layers.Conv2D(f2,(1,1),padding = 'same',activation = 'relu')(x)
  conv13 = tf.keras.layers.Conv2D(f2,(3,3),padding = 'same',activation = 'relu')(conv13)
   
  #5x5 convolution
  conv15 = tf.keras.layers.Conv2D(f3,(1,1),padding = 'same',activation = 'relu')(x)

  conv15 = tf.keras.layers.Conv2D(f3,(5,5),padding = 'same',activation = 'relu')(conv15)
  #3x3 max pooling
  convp = tf.keras.layers.Conv2D(f3,(1,1),padding = 'same',activation = 'relu')(x)

  pool1 = tf.keras.layers.MaxPooling2D((3,3),strides =(1,1),padding = 'same')(convp)
  #concatenate filters
  out = tf.keras.layers.concatenate([conv13,conv15,pool1,conv11])
  return out


'''Creating the model layer by layer'''


def make_seq(Augm, Drop ,N_layers ,pooling ,img_input ,name ):
    if Augm == True:
        # Data Augmentation
        data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
        ])
        x = data_augmentation(img_input)
        x = tf.keras.layers.Conv2D(16,(3,3), padding='same',activation = 'relu')(x)

        x = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',input_shape = (64,64,5))(x)
    else: x = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',input_shape = (64,64,5))(img_input)

    if pooling == True:
        x = tf.keras.layers.MaxPool2D(2,2)(x)

    for a in range(1,N_layers):
        x = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu')(x)

   

    x =  tf.keras.layers.Flatten()(x)
    # Dense layers


    x = tf.keras.layers.Dense(1024,activation = 'relu')(x)
    if Drop == True:
        x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(1024,activation = 'relu')(x)

    x = tf.keras.layers.Dense(151,activation = 'softmax')(x) #generalise it with Nbins

    model = tf.keras.models.Model(img_input,x,name = name)

    return model

''' Function making a model with Pasquet architecture'''

def make_Pasq(Augm,Drop,img_input ,N_layers , name):
    if Augm == True:
        # Data Augmentation
        data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
        ])
        x = data_augmentation(img_input)
        x = tf.keras.layers.Conv2D(16,(3,3), padding='same',activation = 'relu')(x)

        x = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',input_shape = (64,64,5))(x)
    else: x = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',input_shape = (64,64,5))(img_input)



    #x = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu',input_shape = (64,64,5))(img_input)
    x = tf.keras.layers.MaxPool2D(2,2)(x)
    for a in range(1,N_layers):
        x = pasq_inception_module(x,64,64,64)

    x =  tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024,activation = 'relu')(x)
    if Drop == True:
        x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1024,activation = 'relu')(x)

    x = tf.keras.layers.Dense(151,activation = 'softmax')(x)

    model = tf.keras.models.Model(img_input,x,name = name)
    return model

'''Defining the function to make the predictions and produce the metrics for a given model, dataset and Number of classes'''

def show_metrics(model,images, labels, Nbins):
    predictions = model.predict(images)
    # Defining the metrics
    dz = [] # Make a function called make_metrics(images,Nbins)
    mad2 = []
    s = 0
    red = []
    
    # Predicted redshifts
    # For the training dataset
    for i in range(0,len(images)):
        pred_red = np.argmax(predictions[i])/Nbins
        red.append(pred_red)

    # Residuals

    for i in range(0,len(images)):
        dzz = (red[i] - labels[i])/(1+labels[i])
        dz.append(dzz)
    # Prediction bias
    pred_bias = np.mean(dz)

    # MAD deviation 
    med = np.median(dz)
    for a in dz:
        mad1 = (np.abs(a-med))
        mad2.append(mad1)
    MAD = np.median(mad2)
    smad = 1.4826 * MAD

    # Outliers (Defined as in the Pasquet paper)

    for a in dz:
        if a > 5*smad:
            s+=1
    # Outliers fraction
    out_frac = s/len(dz)
    
    return red,dz,pred_bias,smad,out_frac


''' Function that prints metrics '''
def print_stuff(pred_bias,smad,out_frac):
    # Printing the metrics 
    print("The prediction bias is:",pred_bias,end = '\n')
    print("The Mad Deviation is:",smad,end = '\n')
    print("The percentage of outliers is:",out_frac,end = '\n')
    return 0 

''' Function that makes plots '''
def make_plot(labels,red,dz,smad,name = 'model'):

    # Constructing the filenames
    
    file_name_01 = 'Residuals_'+name+'.png'
    file_name_02 = 'Plot' + name + '.png'
    # Plotting the residuals 
    plt.figure()
    plt.hist(dz, bins=100)
    plt.xticks(np.arange(-0.1, 0.1, step=0.02))
    plt.xlim(-0.1,0.1)
    plt.xlabel('Δz')
    plt.ylabel('Relative Frequency')
    plt.title('Δz Distribution')
    plt.savefig(file_name_01, bbox_inches='tight')
    plt.show()

    # Plotting the predictions vs the data 
    fig = plt.figure(figsize = (3,3))
    axis = fig.add_axes([0,0.4,1,1])
    axis2 = fig.add_axes([0,0,1,0.3])
    axis.set_ylabel('Photometric Redshift')
    axis2.set_ylabel('bias (Δz)')
    axis2.set_xlabel('Spectroscopic Redshift')
    axis.plot([0, 0.5], [0, 0.5], 'k-', lw=1)
    axis2.plot([0, 0.5], [smad, smad], 'k-', lw=1)
    axis.plot(labels,red,'ko', markersize=0.3, alpha=0.3)
    #axis.hist2d(labels,red,bins =150)
    axis2.plot(labels,  np.asarray(red) - np.asarray(labels),'ko', markersize=0.3, alpha=0.3)
    
    plt.savefig(file_name_02,dpi = 300,transparent = False,bbox_inches = 'tight')
    plt.show() 

    return

''' Function that makes a model with certain parameters and runs it with specified data'''
def run_model_with_data(model,input_shape,images, labels,test_images,test_labels,Nbins,Nlayers,name = 'model_01',Augmentation = False,Pooling = False, Dropout = False,batch_size = 400, epochs = 20,validation_split = 0.1,monitor = 'loss'):


    # Defining the images' input size
    img_input = tf.keras.layers.Input(shape = input_shape) # 64*64 image in 5 filters

    if model == "seq":
        model = make_seq(Augmentation, Dropout,Nlayers,Pooling,img_input,name=name)
    
    elif model == "Pasquet":
        model = make_Pasq(Augmentation, Dropout, img_input,N_layers=Nlayers,name = name)

    else : print("Choose a proper model type")

    # Show the model
    plot_file_name = name + '.png'
    plot_model(model, to_file=plot_file_name, show_shapes=True, show_layer_names=True)

    # Compile the model

    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(),#learning_rate=1e-3),
              metrics = ['accuracy']
    )

    # Make the classes
    yyy = make_classes(1.,Nbins,labels)
    # Making the callbacks


    callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=5,verbose = 1, restore_best_weights = True)
    # Fit the model

    model_fit = model.fit(images,yyy,                   
                      batch_size = batch_size,
                      epochs = epochs,                                        
                      validation_split = validation_split,
		      verbose = 2,
                      callbacks = [callback] 
                      )
                      
    # Delete the data from memory to be more memory efficient
    del images
    del labels
    gc.collect()
    
    # Save the model
    path_save =  "/sps/lsst/users/atheodor/" + name
    model.save(path_save)

    # Make predictions on the test data and display metrics

    red,dz,pred_bias,smad,out_frac = show_metrics(model,test_images,test_labels,Nbins)
    print_stuff(pred_bias,smad,out_frac)

    # Make plots 
    
    make_plot(test_labels,red,dz,smad,name) 

    

    return
