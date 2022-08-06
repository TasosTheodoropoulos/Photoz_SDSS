import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import gc
from keras.models import load_model
''' modules for scatter density'''
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib import cm
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

def load_data(file_name):
    """ Loads the data from an .npz file
    
    Given the file name and changing the path from /sps/lsst/users/atheodor/
    to the one where your desired file is located returns the images datacubes
    as numpy arrays and the labels (here redshift z).
    
    Arguments:
        file_name(str): name of the file containing the data
    
    Returns:
        numpy array: images
        float: labels
    
    """



    file_path = "/sps/lsst/users/atheodor/" + file_name
    file = np.load(file_path)
    images = file["images"]
    labels = file["label"]
    images = np.asarray(images)
    return images,labels 





def make_classes(red_max,Nbins,labels):
    """
    Makes the bins/classes and putting the data into their respective classes

    Given the maximum redshift considered for the classification,
    the number of bins to be created (i.e. the number of different classes)
    and the labels for the training data it produces the desired amount of 
    classes and puts the labels of the training data into their respective 
    classes.

    Arguments:
        red_max (float): Maximum redshift considered
        Nbins (int): Number of bins considered
        labels (ndarray): The labels of the training data (Here, the redshifts
        of the galaxy images used in training)
    
    Returns:
        yyy(ndarray of ints): Ndarray containing the encoded labels for the
        galaxy images contained in the training dataset.
    
    """


    #red_max = 1. # When the redshift is 1 maybe need to do Nbins + 1 in line 77 
    # Use double values to work properly e.g. 1. not 1
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
  """
  Defining the inception module used in Pasquet et al.
  https://arxiv.org/abs/1806.06607 , inspired by Szegedy et al. https://arxiv.org/abs/1409.4842
  Helpful video on the definition of inception modules:
  https://youtu.be/4DH3JcZqBFI 

  Arguments:
    x: Input layer or output of previous convolutional layer
    f1,f2,f3 (ints): Number of filters in the convolutional layers
  
  Returns:
    out: Output of the inception module, which is the concatenated output
    of the convolutional and pooling layers contained in it.
  
  """

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




def make_seq(Augm, Drop ,N_layers ,pooling ,img_input ,name ):
    """
    Creating the simplest CNN, a sequential model.

    Arguments:
        Augm (Boolean): If True, it adds data augmentation to the model
        input.

        Drop (Boolean): If True, it adds Dropout to the dense part of 
        the CNN (only between the first layers).

        N_layers(int): The Number of Convolutional layers in the model.

        pooling (Boolean): Adds Max Pooling layers after the input layer
        and after every 4 Convolutional layers after the first one.

        img_input (numpy array): Training images as numpy arrays
        (When using jpeg images change the input_shape to (64,64,3))

        name (str): The desired name of the model
    
    Returns:
        model(keras model): The desired sequential keras model ready to 
        be compiled and trained.    
    
    """

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
        # adds max pooling every four layers
        if ((a % 4 == 0) and (pooling == True)):  x = tf.keras.layers.MaxPool2D(2,2)(x)
   

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
    """
        Creating a CNN using a Pasquet-like architecture.
        Essentially a GoogleNet.

        Arguments:
            Augm (Boolean): If True, it adds data augmentation to the model
            input.

            Drop (Boolean): If True, it adds Dropout to the dense part of 
            the CNN (only between the first layers).

            img_input (numpy array): Training images as numpy arrays
            (When using jpeg images change the input_shape to (64,64,3))

            N_layers(int): The Number of Convolutional layers in the model.         

            name (str): The desired name of the model.
        
        Returns:
            model(keras model): The desired GoogleNet keras model ready to 
            be compiled and trained. 

        Todo: Implement MaxPooling like in sequential model
            pooling (Boolean): Adds Max Pooling layers after the input layer
            and after every 4 Convolutional layers after the first one.
    """


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



def show_metrics(model,images, labels, Nbins):
    """
        Makes predictions given the model, test_images,_test_labels and Number of 
        bins, defines and calculates the desired performance metrics for the
        classification (as defined in Pasquet et al.) and returns them.

        Arguments:
            model (keras model): The model we want to predictions with.

            images (numpy array): The images we want to test the model with.
            
            labels (ndarray): The labels for the images used.

            Nbins (int): The number of bins/classes

        Returns:
            red (ndarray): Contains the predicted redshift values for the
            test images. (As predicted value here, we use the most probable value, i.e.
            the value with the highest output).

            dz (ndarray): Residuals for every test image.

            pred_bias (float): The prediction bias, mean of dz.

            smad (float):The MAD deviation.

            out_frac (float): The fraction of outliers.
      
    """


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
    """
        Prints the prediction bias, the MAD deviation and the fraction of outliers.

        Arguments:
            pred_bias (float): The prediction bias.

            smad (float):The MAD deviation.

            out_frac (float): The fraction of outliers.

    """
    # Printing the metrics 
    print("The prediction bias is:",pred_bias,end = '\n')
    print("The Mad Deviation is:",smad,end = '\n')
    print("The percentage of outliers is:",out_frac,end = '\n')
    return 0 

''' Function that makes plots '''
def make_plot(labels,red,dz,smad,name = 'model',lim = 0.4):
    """
        Makes a histogram of the prediction bias and a plot of the estimated Photometric redshift 
        and the Prediction bias versus the Spectroscopic redshift given the labels, red, dz, smad,
        name and lim and it saves both of them in .png files. (Mostly used for debugging purposes)


        Arguments:
            labels (ndarray): The labels for the images used.

            red (ndarray): Contains the predicted redshift values for the test images.

            dz (ndarray): Residuals for every test image.

            smad (float):The MAD deviation.

            name (str): The name of the model.

            lim (float): The limit of the axes of the plot

    """
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
    axis.plot([0, lim], [0, lim], 'k-', lw=1)
    axis.set_xlim([0, lim])
    axis.set_ylim([0,lim])
    axis2.plot([0, lim], [smad, smad], 'k-', lw=1)
    axis2.set_xlim([0, lim])
    axis.plot(labels,red,'ko', markersize=0.3, alpha=0.3)
    #axis.hist2d(labels,red,bins =150)
    axis2.plot(labels,  np.asarray(red) - np.asarray(labels),'ko', markersize=0.3, alpha=0.3)
    
    plt.savefig(file_name_02,dpi = 300,transparent = False,bbox_inches = 'tight')
    plt.show() 

    return

''' Function that makes a model with certain parameters and runs it with specified data'''
def run_model_with_data(model,input_shape,images,labels,test_images,test_labels,Nbins,Nlayers,name = 'model_01',Augmentation = True,Pooling = False, Dropout = False,batch_size = 64, epochs = 20,validation_split = 0.2,monitor = 'loss'):

    """
        Plots the model architecture, compiles, fits and saves the model.
        Prints the metrics and saves the relevant plots for debugging.

        Arguments:
            model (str): The model we want to predictions with. e.g. "seq" or "Pasquet"

            input_shape (tuple): The shape of the input images. e.g. (64,64,5) for SDSS images and (64,64,3) for jpeg SDSS images

            images (numpy array): The images we want to train the model with.
            
            labels (ndarray): The labels for the images used.

            test_images (numpy array): The images we want to test the model with.
            
            test_labels (ndarray): The labels for the test images used.

            Nbins (int): The number of bins/classes.

            N_layers(int): The Number of Convolutional layers in the model.

            name (str): The name of the model.

            Augmentation (Boolean): If True, it adds data augmentation to the model
            input.

            Pooling (Boolean): Adds Max Pooling layers after the input layer 
            and after every 4 Convolutional layers after the first one.
            (does not work with the Pasquet architecture)

            Dropout (Boolean): If True, it adds Dropout to the dense part of 
            the CNN (only between the first layers).

            batch_size (int): Determines the batch size during training.

            epochs (int): Determines the epochs during training.

            validation_split (float): Determines the validation split during training.

            monitor (str): Determines the value to be monitored during training. 
            (used by the EarlyStopping callback function)

        Returns:
            model(keras model): The keras model used.
      
    """

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

    

    return model


def model_pretr_pred(name,test_images, test_labels,Nbins):

    """
        Loads a pretrained model and makes predictions on the given test images.

        Arguments:
            name (str): The name of the model.

            test_images (numpy array): The images we want to test the model with.
            
            test_labels (ndarray): The labels for the test images used.

            Nbins (int): The number of bins/classes.
        
        Returns:
            name (str): The name of the model.

            predictions (list): List containing all the relevant metrics for predictions.

    """

    path = "/sps/lsst/users/atheodor/" + name
    model = load_model(path)
    red,dz,pred_bias,smad,out_frac = show_metrics(model, test_images, test_labels,Nbins)
    predictions = [red,dz,pred_bias,smad,out_frac]

    return name,predictions



def using_mpl_scatter_density(fig, x, y):

    """
        Plots the given data (x and y) in the figure fig using mpl_scatter_density.
        Useful stackoverflow question: https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib

        Arguments:
            fig (str): The name of the model.

            x (ndarray): The data we want to have in the x axis.
            
            y (ndarray): The data we want to have in the y axis.

    """

    viridis = cm.get_cmap("viridis", 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])
    newcolors[:1, :] = white
    viridis_white = ListedColormap(newcolors, name="viridis_white")

    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    #norm = ImageNormalize(vmin=0., vmax=70, stretch=LogStretch())

    density = ax.scatter_density(x, y,vmin=0., vmax=80, cmap= viridis_white)
    fig.colorbar(density, label='Number of points per pixel')

def dens_plot(model_name,test_images, test_labels, max_red = 0.4,Nbins = 180 ):
    """
        Produces a scatter plot, coloured with density of points, of the 
        estimated photometric redshifts produced by the model versus the 
        spectroscopic measurements. In the plot, there is also  the lines
        that define the outliers, the y=x line and a text-box containing 
        the metrics.
        
        Arguments:
            model_name (str): The name of the model.

            test_images (numpy array): The images we want to test the model with.
            
            test_labels (ndarray): The labels for the test images used.

            max_red (float): The maximum redshift considered. (used for the axes limits and the plot of the y=x line)

            Nbins (int): The number of bins/classes.

    """


    name,predictions = model_pretr_pred(model_name,test_images, test_labels, Nbins)
    plot_name = 'Plot_' + model_name
    fig = plt.figure()
    #using_mpl_scatter_density(fig,test_labels,predictions[0])
    outlier_threshold = 5. * predictions[3]
    outliers = predictions[4]*100
    x = np.linspace(0, max_red, 10)
    outlier_upper = x + outlier_threshold * (1 + x)
    outlier_lower = x - outlier_threshold * (1 + x)

    textstr = "\n".join(
                (
                    r"$\sigma_{\mathrm{NMAD}}=%.5f$" % (predictions[3]),
                    r"$\mathrm{f}_{\mathrm{outlier}}=%.2f$" % (outliers) + "%",
                    r"$\langle \frac{\Delta z}{1+z_{\mathrm{spec}}} \rangle=%.5f$"
                    % (predictions[2]),
                )
            )



    fig = plt.figure()
    using_mpl_scatter_density(fig,test_labels,predictions[0])

    plt.xlim(0,max_red)
    plt.ylim(0,max_red)
    # plotting the lines for the outliers
    plt.plot(x, outlier_upper, "k--")
    plt.plot(x, outlier_lower, "k--")

    #plotting the ideal y=x line
    plt.plot([0, max_red], [0, max_red], 'k-', lw=0.5)

    #adding text to the plot
    plt.text(0.01,max_red-0.01,textstr,fontsize=10,verticalalignment="top")

    #adding labels to the axes
    plt.xlabel('Spectroscopic Redshift',fontsize=8)
    plt.ylabel('Photometric Redshift',fontsize=8)

    #saving the plot in a file
    plt.savefig(plot_name,dpi = 600,transparent = False,bbox_inches = 'tight')
    plt.show()

    return
