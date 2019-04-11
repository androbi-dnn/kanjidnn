import numpy as np
import os

from keras.models import Model
from keras.optimizers import Optimizer 
from keras.optimizers import Adam 
from keras.callbacks import CSVLogger

#import dataetl
#from kanjidnn import ModelBag

class DNNTool():

    def __init__(self):
        self.optimizer = None
        self.model = None
        self.model_compiled = False
        self.data = None
        self.data_loaded = False
        self.n_output = 0
        self.csv_logger = None
        self.input_shape = None
        self.set_optimizer()

    def set_optimizer(self, optimizer = Adam(lr=1e-4)):
        """Set the Keras optimizer to use for training."""
        if not isinstance(optimizer, Optimizer):
            raise TypeError('optimizer must be of type keras.optimizers.Optimizer')
        self.optimizer = optimizer
        self.model_compiled = False
        print("Optimizer",type(self.optimizer).__name__,"set. Learning rate:",self.optimizer.lr)

    def set_model(self, model):
        """Set the Keras model to use for training."""
        if not isinstance(model, Model):
            raise TypeError('model must be of type keras.models.Model')
        self.model = model
        self.model_compiled = False

    def set_data_source(self, data):
        """Set an instance of DataETL as data source."""
        # TODO: add type check
        self.data = data

    def compile(self):
        """Compile the Keras model."""
        print("Compiling model ..")
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer, 
            metrics=['accuracy'])
        self.model_compiled = True

    def load_data(self,only_first=False):
        """Load data from data source."""
        if self.data is None:
            print("Error: data source has not been set. Use set_data_source() to specify data source.")
            return
        print ("Loading training and test data ..")
        self.n_output, self.input_shape  = self.data.load_data(only_first=only_first)
        self.data_loaded = True
    
    def info(self):
        """Print information about current state."""
        print("------- Info --------")
        if self.data is not None:
            print("Data source:",type(self.data).__name__)
            if self.data_loaded:
                print ("Training size: ", self.data.x_train.shape[0])
                print ("Test size    : ", self.data.x_test.shape[0])
                print ("Classes      : ", self.n_output)
            else:
                print("Data not loaded.")
        else:
            print("Data source not set.")
        
        if self.model is not None:
            print("Model type:",self.model.name)
            if self.model_compiled:
                self.model.summary()
            else:
                print("Model not yet compiled.")
        else:
            print("Model type not set.")

    def train(self, epochs=10, batch_size=32, logger_fn=None):
        """Train the model with training data."""
        print ("Start training ..")
        callbacks = []
        if logger_fn is not None:
            callbacks.append(CSVLogger(logger_fn, append=True, separator=';'))
        self.model.fit(self.data.x_train, self.data.y_train, epochs=epochs, batch_size=batch_size,  callbacks=callbacks) 

    def evaluate(self, batch_size=32):
        """Evaluate the model with test data."""
        score, acc = self.model.evaluate(self.data.x_test, self.data.y_test, batch_size=batch_size, verbose=0)
        print ("Test size: ", self.data.x_test.shape[0])
        print ("Test Score: ", score)
        print ("Test Accuracy: ", acc)

    def save_weights(self, name):
        self.model.save_weights(name)

    def load_model_weights(self, name, by_name=False):
        self.model.load_weights(name,by_name=by_name)

if __name__ == '__main__':
    dnn = DNNTool()
    #dnn.set_data_source(DataETL8B())
    #dnn.load_data(only_first=True)
    #bag = ModelBag()
    #dnn.set_model(bag.mobile_net_64(classes = dnn.n_output, input_shape = dnn.input_shape))
    dnn.info()
    #dnn.compile()
    #dnn.train(epochs=1, batch_size=32, logger_fn='log.csv')
    #dnn.save_weights('gdrive/My Drive/Data/Vgg/weights_etl9b_10.h5')
