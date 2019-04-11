import numpy as np
import os, sys
import struct
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend
from keras.utils import np_utils
from abc import ABC, abstractmethod

class DataETL(ABC):
    """Abstract ETL data source base class."""
    def __init__(self,etl_path='ETLC'):
        self.ETL_PATH = etl_path
        self.WRITERS=0
        self.RECLENGTH=0
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.inv_map = None
        super().__init__()
    
    @abstractmethod
    def read_record(self, f):
        pass
    @abstractmethod
    def get_dataset(self, n_data):
        pass
    @abstractmethod
    def load_data(self, test_size=0.2):
        pass

    # helper function to read a record given a format for record structure
    def read_etl_record(self, format, width, height, f):
        s = f.read(self.RECLENGTH)
        r = struct.unpack(format, s)
        # 1 bit per pixel
        i1 = Image.frombytes('1', (width, height), r[3], 'raw')
        img_out = r + (i1,)
        return img_out

    # go to a given record in file
    def goto_etl_record(self, f, n, skip=0):
        offset =  n * self.WRITERS * self.RECLENGTH + skip*self.RECLENGTH
        f.seek(offset)

    # relabel y_train from uint16 -> 1..nb_classes
    def relabel(self):
        unique_labels = list(set(self.y_train.flatten()))
        nb_classes = len(unique_labels)
        labels_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
        # store a map that permits to recover original label
        self.inv_map = {v: k for k, v in labels_dict.items()}
        self.y_train[:,0] = np.array([labels_dict[l] for l in self.y_train.flatten()], dtype=np.uint16)
        return nb_classes
    
    # helper to get test and training sets. Call relabel() before to obtain nb_classes
    # This is a memory optimized version of train_test_split that was necessary for 
    # the ETL 9B dataset on smaller RAM machines.
    def shuffle_and_split(self, nb_classes, test_size=0.2, random_state = 42):
        n_size = self.x_train.shape[0]
        assert (n_size == self.y_train.shape[0])
        # TODO add some sanity tests about n_split/test_size
        n_split = int((1.0-test_size)*self.x_train.shape[0])
        
        np.random.seed(random_state)
        self.shuffle_in_unison_scary()

        # split off last part
        self.x_test = np.copy(self.x_train[n_split:n_size,:,:,:])
        self.y_test = np.copy(self.y_train[n_split:n_size,:])

        # resize first part
        self.x_train.resize([n_split,64,64,1])
        self.y_train.resize([n_split,1])
        
        return

    # see https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    def shuffle_in_unison_scary(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self.x_train)
        np.random.set_state(rng_state)
        np.random.shuffle(self.y_train)

    def check_if_file_exists(self, filename):
        exists = os.path.isfile(filename)
        return exists


# ETL8 binalized data, see http://etlcdb.db.aist.go.jp/?page_id=2461
# we skip hiragana, so we have 881 kanji from 160 different writers
class DataETL8B(DataETL):
    def __init__(self):
        super().__init__()
        self.DATASET_DIR = self.ETL_PATH + '/ETL8B'
        self.DATASET_PREFIX = 'ETL8B2C'
        if not os.path.isdir(self.DATASET_DIR):
            print('Warning: directory',self.DATASET_DIR,'does not exist. Set DATASET_DIR')

        self.WIDTH = 64
        self.HEIGHT = 64
        self.RECLENGTH = 512
        self.WRITERS = 160
        self.NUM_CHARS = 881
        self.NUM_DATASETS = 3
        self.SAVE_WRITER = 0 # set to negative to skip image writing

    # read a single record
    def read_record(self, f):
        # 1-2	2	Integer	Serial Sheet Number
        # 3-4	2	Binary	JIS Kanji Code (JIS X 0208)
        # 5-8	4	ASCII	JIS Typical Reading ( ex. ‘AI.M’)
        # 9-512	504	Packed	Binary image of 64 x 63 = 4032 pixels
        return self.read_etl_record('>2H4s504s', self.WIDTH, self.HEIGHT-1, f)
 
    # read a single dataset
    def get_dataset(self, n_data):
        start_record = 0  
        if n_data == 1: # skip hiragana in first dataset
            start_record = 75
        max_records = 320
        if (n_data == self.NUM_DATASETS): # less recs in last
            max_records = 316
        records = range(start_record, max_records)

        new_img = Image.new('1', (self.WIDTH, self.HEIGHT))
        filename = self.DATASET_DIR + '/' + self.DATASET_PREFIX + str(n_data)
        print("reading",filename)

        X = []
        Y = []
        if self.SAVE_WRITER >= 0:
            save_img = Image.new('1', (64*18, 64*18))

        with open(filename, 'rb') as f:
            for n_rec in records:
                self.goto_etl_record(f, n_rec,skip=1)
                for i in range(self.WRITERS):
                    # read, paste into img and invert
                    r = self.read_record(f)
                    if i==self.SAVE_WRITER:
                        save_img.paste(r[-1], (64*(n_rec%18), 64*(n_rec//18)))
                    new_img.paste(r[-1], (0, 0))
                    iI = Image.eval(new_img, lambda x: not x)
                    # append as numeric data to image X and labels Y
                    out_data = np.asarray(iI.getdata(), dtype=np.uint8).reshape(self.WIDTH, self.HEIGHT)                        
                    X.append(out_data)
                    Y.append(r[1])

        if self.SAVE_WRITER >= 0:
            iI = Image.eval(save_img, lambda x: not x)
            fn = 'ETL8B2_{:03d}.png'.format(n_data)
            iI.save(fn, 'PNG')
        return np.asarray(X, dtype=np.uint8), np.asarray(Y, dtype=np.uint16)

    # read all sets (or only_first) into self.x_train and self.y_train
    def load_data(self, test_size=0.2, only_first=False):
        num_chars=[320-75,320,316]
        n_sets = self.NUM_DATASETS+1
        if only_first:
            n_sets = 2
            num_chars=num_chars[0:1]
        # allocate characters and labels
        self.x_train = np.empty([sum(num_chars)*self.WRITERS, 64, 64, 1], dtype=np.uint8)
        self.y_train = np.empty([sum(num_chars)*self.WRITERS, 1], dtype=np.uint16)
        # read sets
        try:
            for i in range(1, n_sets):
                lower = sum(num_chars[0:i-1])*self.WRITERS
                upper = sum(num_chars[0:i])*self.WRITERS
                self.x_train[lower:upper,0:64,0:64,0],self.y_train[lower:upper,0] = self.get_dataset(i)
        except FileNotFoundError as not_found:
            print('Error: File ',not_found.filename,'not found')
            return None, None

        nb_classes = self.relabel()
        self.shuffle_and_split(nb_classes)
        input_shape = (self.x_train.shape[1], self.x_train.shape[2], 1)
        return nb_classes, input_shape

# ETL9 binalized data, see http://etlcdb.db.aist.go.jp/?page_id=1711
class DataETL9B(DataETL):
    def __init__(self):
        super().__init__()
        self.DATASET_PREFIX = self.ETL_PATH + '/ETL9B/ETL9B_'
        self.WIDTH = 64
        self.HEIGHT = 64
        self.RECLENGTH = 576
        self.NUM_CHARS = 3036 -71
        self.WRITERS = 40   # This is per set only. Total = 5*40 writers
        self.NUM_DATASETS = 5
        self.SAVE_WRITER = 0 # set to negative to skip image writing

    # read a single record
    def read_record(self, f):
        # 1-2	2	Integer	Serial Sheet Number
        # 3-4	2	Binary	JIS Kanji Code (JIS X 0208)
        # 5-8	4	ASCII	JIS Typical Reading ( ex. ‘AI.M’)
        # 9-512	504	Packed	Binary image of 64 x 63 = 4032 pixels
        # 513-576	64		(uncertain)
        return self.read_etl_record('>2H4s504s64s', self.WIDTH, self.HEIGHT-1, f)

    # read a single dataset
    def get_dataset(self, n_data):
        # rec numbers from 0 .. max - 1
        start_record = 71
        max_records = 3036

        records = range(start_record, max_records)        
        new_img = Image.new('1', (self.WIDTH, self.HEIGHT))

        filename = self.DATASET_PREFIX + str(n_data)
        print("reading",filename)

        X = []
        Y = []
        if self.SAVE_WRITER >= 0:
            save_img = Image.new('1', (64*33, 64*92))

        if not self.check_if_file_exists(filename):
            print("file",filename,"not found.")
            return np.asarray(X, dtype=np.uint8), np.asarray(Y, dtype=np.uint16)

        with open(filename, 'rb') as f:
            self.goto_etl_record(f,0,1)
            for i in range(self.WRITERS):
                # skip hiragana
                for j in range(71):
                    r = self.read_record(f)
                for n_rec in records:
                    # read, paste into img and invert
                    r = self.read_record(f)
                    if i==self.SAVE_WRITER:
                        save_img.paste(r[-1], (64*(n_rec%33), 64*(n_rec//33)))
                    new_img.paste(r[-1], (0, 0))
                    #if r[1]>=12352 and r[1]<=12447:
                    #print(n_rec,i,r[1],r[1]>=12352 and r[1]<=12447)
                    iI = Image.eval(new_img, lambda x: not x)
                    # append as numeric data to image X and labels Y
                    outData = np.asarray(iI.getdata(), dtype=np.uint8).reshape(self.WIDTH, self.HEIGHT)                        
                    X.append(outData)
                    Y.append(r[1])

        if self.SAVE_WRITER >= 0:
            iI = Image.eval(save_img, lambda x: not x)
            fn = 'ETL9B_{:03d}.png'.format(n_data)
            iI.save(fn, 'PNG')

        return np.asarray(X, dtype=np.uint8), np.asarray(Y, dtype=np.uint16)

    def load_data(self, test_size=0.2, only_first=False):
        n_sets = self.NUM_DATASETS+1
        if only_first:
            n_sets = 2
        # characters has a size of 2.4 Gb. Allocate now and don't copy around
        size = self.NUM_CHARS*self.WRITERS*(n_sets-1)
        self.x_train = np.empty([size, 64, 64, 1], dtype=np.uint8)
        self.y_train = np.empty([size, 1], dtype=np.uint16)
        for i in range(1, n_sets):
            lower = (i-1)*self.NUM_CHARS*self.WRITERS
            upper = (i)*self.NUM_CHARS*self.WRITERS
            self.x_train[lower:upper,0:64,0:64,0],self.y_train[lower:upper,0] = self.get_dataset(i)

        nb_classes = self.relabel()
        self.shuffle_and_split(nb_classes)
        input_shape = (self.x_train.shape[1], self.x_train.shape[2], 1)
        return nb_classes, input_shape

if __name__ == '__main__':
    # typical usage:
    data = DataETL9B()
    nb_classes, input_shape = data.load_data(only_first=True)
    print("classes:",nb_classes)
    print("input_shape:",input_shape)
    print("x_train.shape:", data.x_train.shape)
    print("y_train.shape:", data.y_train.shape)
    print("training:",data.x_train.shape[0], "testing:", data.x_test.shape[0], "total:",
            data.x_train.shape[0]+data.x_test.shape[0])
