""" serve up some ancient scroll data """

import os
import sys
import numpy as np
from scipy.ndimage import measurements as me

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
        return train_data['x'][...,0,np.newaxis], train_data['y'], train_data['m'][...,0,np.newaxis]
    elif subset=='test':
        test_data = unpickle(os.path.join(data_dir,'letters_test.pkl'))
        return test_data['x'][...,0,np.newaxis], test_data['y'], test_data['m'][...,0,np.newaxis]
    else:
        raise NotImplementedError('subset should be either train or test')

def get_orientations(ms):
    o = np.zeros(len(ms), dtype=np.int32) # orientations
    for i, m in enumerate(ms):
        m = m[:,:,0]
        y,x = me.center_of_mass(m)
        if np.isnan(x) or np.isnan(y): 
            continue
        # center coordinates
        y -= 15.5
        x -= 15.5
        # fill o with optimal orientation for each mask, to maximize exposure
        # of known information (1s in the mask) to PixelCNN
        if y>=0 and x>=0:
            if y>x:
                o[i] = 2 # 2 rotations
            elif y<=x:
                o[i] = 7 # flip + 3 rotations
        elif y>=0 and x<0:
            if y>=-x:
                o[i] = 6 # flip + 2 rotations
            elif y<-x:
                o[i] = 3 # 3 rotations
        elif y<0 and x<0:
            if y>=x:
                o[i] = 5 # flip + 1 rotations
            elif y<x:
                o[i] = 0 # no flips or rotations, this is optimal
        elif y<0 and x>=0:
            if y>=-x:
                o[i] = 4 # just flip no rotation needed
            if y<-x:
                o[i] = 1 # one rotation
    return o

class DataLoader(object):
    """ an object that generates batches of data for training """

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False, rotation=None, test=False):
        """ 
        - data_dir is location where to store files
        - subset is train|test 
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        - self.test indicates if we are in adaptive rotation mode with single model per rotation or not
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
        
        self.size = len(self.data)
        
        if self.test and self.rotation is not None:
            y = get_orientations(self.masks)
            inds = (y == self.rotation)
            self.data = self.data[inds]
            self.masks = self.masks[inds]
            
        # padding!!
        csz = len(self.data)
        sz = csz + (self.batch_size - csz % self.batch_size )
        psz = [sz] + list(self.masks.shape[1:])

        zmasks = np.cast[self.masks.dtype](np.zeros(psz))
        zmasks[:csz] = self.masks
        self.masks = zmasks
        
        zdata = np.cast[self.data.dtype](np.zeros(psz))
        zdata[:csz] = self.data
        self.data = zdata
        
        self.size = csz
        print('loaded {} samples in orientation {}, totalling with {} padding'.format(
                self.size, self.rotation, sz))
        
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def get_num_labels(self):
        return 8 # 4 rotations * 2 possible flips

    def set_batch_size(self, n):
        self.batch_size = n
        return

    def get_batch_size(self):
        return self.batch_size
    
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

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        m = self.masks[self.p : self.p + n]
        self.p += self.batch_size
        
        return x,m
    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)