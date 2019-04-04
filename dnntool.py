import numpy as np
import os
import warnings

from keras.models import Sequential
from keras.optimizers import Adam 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import CSVLogger

from keras import initializers
from keras import backend
from keras import layers
from keras import models

from dataetl import DataETL8B, DataETL9B
from modelbag import ModelBag

class DNNTool():

    def __init__(self):
        self.opt = None
        self.model = None
        self.model_compiled = False
        self.data = None
        self.data_loaded = False
        self.n_output = 0
        self.input_shape = None
        self.set_optimizer()

    def set_optimizer(self, opt=Adam(lr=1e-4)):
        self.opt = opt

    def set_model(self, model):
        self.model = model

    def set_data(self, data):
        self.data = data

    def prepare(self):
        print("Compiling model ..")
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.opt, metrics=['accuracy'])
        self.model_compiled = True

    def load_data(self):
        print ("Loading training and test data ..")
        self.n_output, self.input_shape  = self.data.load_data(only_first=self.only_first)
        print ("Training size: ", self.data.x_train.shape[0])
        print ("Test size: ", self.data.x_test.shape[0])
        print ("Classes: ", self.n_output)
        self.data_loaded = True
    
    def info(self):
        if self.data not None:
            print("Data type set:",type(self.data).__name__)
            if self.data_loaded:
                print("Data loaded.")
                print ("Training size: ", self.data.x_train.shape[0])
                print ("Test size: ", self.data.x_test.shape[0])
                print ("Classes: ", self.n_output)
        else:
            print("Data type not set.")
        
        if self.model not None:
            print("Model type set:",type(self.model).__name__)
            self.model.summary()
        else:
            print("Model type not set.")

 
    def train(self, epochs=10, batch_size=32, sample_interval=200):
        print ("Start training ..")
        csv_logger = CSVLogger('gdrive/My Drive/Data/Vgg/log.csv', append=True, separator=';')
        self.model.fit(self.data.x_train, self.data.y_train, epochs=epochs, batch_size=batch_size,  callbacks=[csv_logger]) 

        score, acc = self.model.evaluate(self.data.x_test, self.data.y_test, batch_size=batch_size, verbose=0)

        print ("Training size: ", self.data.x_train.shape[0])
        print ("Test size: ", self.data.x_test.shape[0])
        print ("Test Score: ", score)
        print ("Test Accuracy: ", acc)

    def save_weights(self, name):
        self.model.save_weights(name)

    def load_model_weights(self, name):
        self.model.load_weights(name)

if __name__ == '__main__':
    dnn = DnnTool()
    dnn.set_data(DataETL9B())
    bag = ModelBag()
    dnn.set_model(bag.mobile_net_64(classes = self.n_output, input_shape = self.input_shape))
    dnn.prepare()
    dnn.train(epochs=10, batch_size=32, sample_interval=200)
    dnn.save_weights('gdrive/My Drive/Data/Vgg/weights_etl9b_10.h5')
