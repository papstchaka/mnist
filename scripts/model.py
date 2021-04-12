'''
Script implements a standard Neural Network, using Triplet Loss (intended to help during feature reduction processes)
'''

## Imports
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

class NN:

    def __init__(self, model:object, X:np.array, y:np.array, test_X:np.array, test_y:np.array, batch_size:int = 32, lr:float = 1e-3, epochs:int = 100, patience:int = 3, verbose:int = 1) -> None:
        '''
        constructor of class. 
        Parameters:
            - model: the used model for the riplet loss [Tensorflow.model = object]
            - X: data containing all train features [numpy.array]
            - y: data containing the train labels [numpy.array]
            - test_X: data containing all test features [numpy.array]
            - test_y: data containing the test labels [numpy.array]
            - batch_size: desired size of batches [Integer, default = 32]
            - lr: desired learning rate [Float, default = 1e-3]
            - epochs: desired number of epochs [Integer, default = 100]
            - patience: patience before training gets stopped by EarlyStopping [Integer, default = 3]
            - verbose: how detailed the training process shall be printed [Integer, default = 1]
        Initializes:
            - lr
            - batch_size
            - epochs
            - X, y, test_X, test_y
            - patience
            - model
            - verbose
        Returns:
            - None
        '''
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.X, self.y, self.test_X, self.test_y = X, y, test_X, test_y
        self.patience = patience
        ## set up model
        self.model =  model
        ## compile model with learning rate and triplet loss
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss=tfa.losses.TripletSemiHardLoss())
        self.verbose = verbose
        
    def train(self) -> np.array:
        '''
        trains the model
        Parameters:
            - None
        Returns:
            - None
        '''
        ## add EarlyStopping callback
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor = "loss", patience = self.patience, restore_best_weights = True)
        ]
        ## train model
        self.model.fit(x=self.X, y=self.y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, validation_data = (self.test_X, self.test_y), callbacks = callbacks)

    def predict(self, X:np.array) -> np.array:
        '''
        predicts values using model
        Parameters:
            - X: data to predict [numpy.array]
        Returns:
            - X_pred: predicted data [numpy.array]
        '''
        X_pred = self.model.predict(self.X)
        return X_pred