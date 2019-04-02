import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization

from DataETL import DataETL9B

class TrainVgg():

    def __init__(self, data):
        self.input_shape = None
        self.only_first = False
        self.data = data
        self.load_data()
        self.adam = Adam(lr=1e-4)
        self.model = self.M7_1_9()
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.adam, metrics=['accuracy'])
        self.model.summary()

    def load_data(self):
        print ("Loading training and test data ..")
        nb_classes = self.data.load_data(only_first=self.only_first)
        self.n_output = nb_classes
        print ("Training size: ", self.data.x_train.shape[0])
        print ("Test size: ", self.data.x_test.shape[0])
        print ("Classes: ", self.n_output)

    def M7_1_9(self):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), padding='same',
                        input_shape=self.input_shape, kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',
                        kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), padding='same',
                        kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same',
                        kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(4096, activation="relu", kernel_initializer='he_normal'))
        model.add(Dropout(0.5))

        model.add(Dense(4096, activation="relu", kernel_initializer='he_normal'))
        model.add(Dropout(0.5))

        model.add(Dense(self.n_output, activation="softmax"))

        return model

    def train(self, epochs=10, batch_size=32, sample_interval=200):

        self.model.fit(self.data.x_train, self.data.y_train, epochs=epochs, batch_size=batch_size) 

        score, acc = self.model.evaluate(self.data.x_test, self.data.y_test, batch_size=batch_size, verbose=0)

        print ("Training size: ", self.data.x_train.shape[0])
        print ("Test size: ", self.data.x_test.shape[0])
        print ("Test Score: ", score)
        print ("Test Accuracy: ", acc)

        #save_model_weights('weights/weights_out.h5', model)

    def save_weights(self, name):
        self.model.save_weights(name)

    def load_model_weights(self, name):
        self.model.load_weights(name)

if __name__ == '__main__':
    vgg = TrainVgg(DataETL8B())
    vgg.train(epochs=10, batch_size=32, sample_interval=200)
    vgg.save_weights('weights/weights_out.h5')