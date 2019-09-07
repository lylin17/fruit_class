import os
import pickle

from keras import metrics
from keras.applications.vgg16 import VGG16
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Activation,Dense,Dropout,Flatten,BatchNormalization
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import wget

class Data:
    '''Class to get data ready for processing
    
    Args:
        fruit_classes(list): list of fruit classes. Must be in ['apples', 'oranges', 'pears']
        
    Attributes:
        fruit_classes(list): list of fruit classes. Must be in ['apples', 'oranges', 'pears']
        x_apples(np array): 350 x 244 x 244 x 3 array for apple images if 'apples' specified in fruit_classes else None
        x_oranges(np array): 350 x 244 x 244 x 3 array for orange images if 'oranges' specified in fruit_classes else None
        x_pears(np array): 350 x 244 x 244 x 3 array for pear images if 'pears' specified in fruit_classes else None        
    '''
    
    def __init__(self, fruit_classes= ['apples','oranges','pears']):
        self.fruit_classes = fruit_classes
        
        self.x_apple, self.x_orange, self.x_pear = None, None, None
    
    def get_data(self):
        '''Use wget to download pickle files from google

        Return:
            None, but pickle file of fruit classes download into ./data
        '''

        #setup data directory in current directory if not present
        if not os.path.isdir('./data'):
            os.mkdir('./data')

        #apples
        if 'apples' in self.fruit_classes:
            path  = './data/x_apples.pkl'
            if not os.path.exists(path):
                print('start download x_apples.pkl...', end='')
                #wget.download('https://dl.dropboxusercontent.com/s/ovah81j9eqicxz4/x_apples.pkl', out=path, bar=None)
                wget.download('https://drive.google.com/uc?export=download&id=1rFDkhfChmKU-9X_UcqdpFwTLFs6mm-LR', out=path, bar=None)
                print('done')

        #oranges
        if 'oranges' in self.fruit_classes:
            path  = './data/x_oranges.pkl'
            if not os.path.exists(path):
                print('start download x_oranges.pkl...', end='')
                #wget.download('https://dl.dropboxusercontent.com/s/737qx4tjhp2foha/x_oranges.pkl', out=path, bar=None)
                wget.download('https://drive.google.com/uc?export=download&id=1bLulvDKAMtPgALrcdH3HfzBwLOFItR4s', out=path, bar=None)
                print('done')

        #pears
        if 'pears' in self.fruit_classes:
            path  = './data/x_pears.pkl'
            if not os.path.exists(path):
                print('start download x_pears.pkl...', end='')
                #wget.download('https://dl.dropboxusercontent.com/s/vb7bbtnciqjg7gv/x_pears.pkl', out=path, bar=None)
                wget.download('https://drive.google.com/uc?export=download&id=1-nerhTEFRPABATA1gP3EJEDsok8tzdJl', out=path, bar=None)
                print('done')

    def unpickle(self,fruit_classes=['apples','oranges','pears']):
        '''Unpickle to get np array for each data set'''
        
        if 'apples' in self.fruit_classes:
            self.x_apple = pickle.load(open('./data/x_apples.pkl', 'rb'))
            
        if 'oranges' in self.fruit_classes:
            self.x_orange = pickle.load(open('./data/x_oranges.pkl', 'rb'))
        
        if 'pears' in self.fruit_classes:
            self.x_pear = pickle.load(open('./data/x_pears.pkl', 'rb'))
                
    def format_split(self, num_images, test_size=0.3):
        '''Format and split data into train and val sets
        
        Args:
            num_images(int): first num_image images to be output
            test_size (float): proportion of data to put in test set
            
        Returns:
            x_train(array): x data for train set
            x_val(array): x data for val set
            y_train(array): y label for train set in one-hot encoding
            y_val(array): y label for val set in one-hot encoding
        '''
        
        num_classes = len(self.fruit_classes)
        
        # Put together a dataset with labels
        mapping = {'apples': self.x_apple, 'oranges': self.x_orange, 'pears': self.x_pear}
        fruit_data = []
        for i in self.fruit_classes:
            fruit_data += [mapping[i][:num_images,: ,: ,:]] 
        
        x = np.concatenate(fruit_data,axis=0)
        
        y = []
        for j in range(num_classes):
            y += [j]*num_images

        #Train-test split and format labels for keras multiclass classification
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size, random_state=42)
        y_train = to_categorical(y_train,num_classes,'float32')
        y_val = to_categorical(y_val,num_classes,'float32')
        
        return x_train, x_val, y_train, y_val
    
class Model:
    '''Class to set model and fit to data
    
    Args:
        initializer(str): to be used to initialize hidden units
        activation_func(str): to be used as activation function for all FC layers except the last layer
        regparam(str): L2 regularization parameters for all FC layers
        last_layer_activation(str): activation function for the last layer. 'softmax' for multiclass, 'sigmoid' for binary
        loss_func(str): loss function used for back propagation
        metric(str): metric used for evaluation
        
    Attributes:
        model(keras object): list of fruit classes. Must be in ['apples', 'oranges', 'pears'].
        modelfit(keras object): fitted model
        datagen(keras object): Keras image generator for data augmentation
        lrsch(keras object): Keras learning rate scheduler
        checkpoint(keras object): Keras model checkpoint
    '''
       
    def __init__(self, initializer='he_normal', activation_func='relu', regparam=5e-3, last_layer_activation='softmax',
             loss_func= 'categorical_crossentropy', metric = 'accuracy'):
        
        # Load pretrained VGG-16 model without FC layer
        model = VGG16(weights = "imagenet", include_top=False, input_shape = (244, 244, 3))

        # Freeze all pre-trained layers
        for layer in model.layers:
            layer.trainable = False

        #Define FC layers 
        Initializer = initializer
        activation = Activation(activation_func)
        Regularizer = l2
        regparam = regparam    

        self.model = Sequential()
        self.model.add(model)

        self.model.add(Flatten())

        self.model.add(Dense(64,kernel_initializer=Initializer,kernel_regularizer=Regularizer(regparam)))
        self.model.add(BatchNormalization(momentum=0.9))
        self.model.add(activation)
        self.model.add(Dropout(0.7,seed=123))

        self.model.add(Dense(3,kernel_initializer=Initializer))
        self.model.add(Activation(last_layer_activation))
        
        # Set optimizers and compile model
        adam = Adam(beta_1=0.9, beta_2=0.999, decay=0)
        self.model.compile(loss=loss_func,
                           optimizer=adam,
                           metrics = [metric])
        
        self.modelfit = None
        self.datagen = None
        self.lrsch = None
        self.checkpoint = None        
        
    def model_utils(self, rotation_range=10, horizontal_flip=True, 
                    model_save='model.h5', monitor='val_loss', mode='min', period=1):
    
        '''Model utils for model training
        
        Args:
            rotation_range(int): rotation range for data augmentation
            horizontal_flip(bool): whether to include horizontal flip for data augmentation
            model_save(str): file name for model file to be saved at checkpoint
            monitor(str): metric to monitor to determine best model
            mode(str): mode for monitor metric
            period(int): number of epochs to check best checkpoint            
        '''         
        
        #Define data generator for data augmentation
        self.datagen = ImageDataGenerator(rotation_range=rotation_range, horizontal_flip=horizontal_flip)

        #Learning rate schedule for stochastic gradient descent with restarts (cosine annealing)   
        
        def lr_sch(epoch):   
            min_lr = 1e-5
            max_lr = 1e-3
            # number of epochs to restart
            restart = 5 

            lrate = min_lr + 0.5*(max_lr - min_lr) * (1+np.cos(((epoch - (epoch//restart * restart))/ restart) * np.pi))      
            return lrate

        self.lrsch = LearningRateScheduler(lr_sch)

        #Save best model file at checkpoint 
        self.checkpoint = ModelCheckpoint(model_save, monitor=monitor, save_best_only=True, save_weights_only=False, mode=mode, period=period)
        
    def fit(self, x_train, y_train, x_val, y_val, batch_size, epochs, verbose):
        '''Fit model to data
        
        Args:
            datagen(keras object): keras data generator for data augmentation
            lrsch(keras object): keras learning rate scheduler
            checkpoint(keras object): save best checkpoint as model.h5 file
            batch_size(int): size of batch for training
            epoches(int): number of epoches to train
            verbose(int): verbosity of keras training  
        '''        
        self.modelfit = self.model.fit_generator(self.datagen.flow(x_train, y_train, batch_size),
                                            steps_per_epoch = x_train.shape[0] / batch_size, 
                                            epochs=epochs,
                                            verbose=1,
                                            validation_data = (x_val,y_val),
                                            callbacks=[self.lrsch, self.checkpoint] 
                                            )


    def evaluate(self, x_val, y_val, model_save='model.h5'):
        '''Evaluate saved model.h5 using x_val and y_val
        
        Args:
            model_save(str): file name for model file to be saved at checkpoint
            
        Returns:
            evaluate(tuple): (validation loss, validation accurarcy)
        
        '''
        
        model_saved = load_model('model.h5')
        evaluate = model_saved.evaluate(x_val, y_val)
        
        return evaluate


    
         