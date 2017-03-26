""" serve up some ancient scroll data """

import os
import sys
import numpy as np

def unpickle(file):
    fo = open(file, 'rb')
    if (sys.version_info >= (3, 0)):
        import pickle
        d = pickle.load(fo)
    else:
        import cPickle
        d = cPickle.load(fo)
    fo.close()
    return {'x': d['data'], 'y': d['labels'], 'm': d['masks']}

def load(data_dir, subset='train'):
    if subset=='train':
        train_data = unpickle(os.path.join(data_dir,'letters_train.pkl'))
        return train_data['x'], train_data['y'], train_data['m']
    elif subset=='test':
        test_data = unpickle(os.path.join(data_dir,'letters_test.pkl'))
        return test_data['x'], test_data['y'], test_data['m']
    else:
        raise NotImplementedError('subset should be either train or test')


class DataLoader(object):
    """ an object that generates batches of data for training """

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False, rotation=None, test=False):
        """ 
        - data_dir is location where to store files
        - subset is train|test 
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels
        self.rotation = rotation
        self.test = test

        # create temporary storage for the data, if not yet created
        if not os.path.exists(data_dir):
            print('creating folder', data_dir)
            os.makedirs(data_dir)

        # load CIFAR-10 training data to RAM
        self.data, self.labels, self.masks = load(data_dir, subset=subset)
        
        if self.test and self.rotation is not None:
            inds = self.labels == self.rotation
            self.data = self.data[inds]
            self.labels = self.labels[inds]
            self.masks = self.masks[inds]
        
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def get_num_labels(self):
        return np.amax(self.labels) + 1

    def set_batch_size(self, n):
        self.batch_size = n
        return

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]
            self.masks = self.masks[inds]
            self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        y = self.labels[self.p : self.p + n]
        m = self.masks[self.p : self.p + n]
        self.p += self.batch_size
        
        # (randomly) rotate batch
        if self.rotation is None:
            k = np.random.randint(4)
            y.fill(k)
        else:
            k = self.rotation
        x = np.rot90(x, k=k, axes=(1,2))
        m = np.rot90(m, k=k, axes=(1,2))
        
        return x,y,m
    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)