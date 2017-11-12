""" serve up some ancient scroll data """

import os
import sys
import numpy as np
from scipy.ndimage import measurements as me
from data.caching import load_data


def load(data_dir, subset='train'):
    if subset=='train':
        return load_data(os.path.join(data_dir, 'train'))
    elif subset=='val':
        return load_data(os.path.join(data_dir, 'val'))
    elif subset=='test':
        return load_data(data_dir)
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
                o[i] = 1 # one rotation
            if y<-x:
                o[i] = 4 # just flip no rotation needed
    return o

class DataLoader(object):
    """ an object that generates batches of data for training """

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False, rotation=None, single_angle=False, pad=False):
        """
        - data_dir is location where to store files
        - subset is train|val|test
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        - self.test indicates if we are in adaptive rotation mode with single model per rotation or not
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels
        self.rotation = rotation
        self.single_angle = single_angle

        # create temporary storage for the data, if not yet created
        if not os.path.exists(data_dir):
            print('creating folder', data_dir)
            os.makedirs(data_dir)

        # load training data to RAM
        self.data, self.masks = load(data_dir, subset=subset)

        self.size = len(self.data)

        # if using data from only a single angle of rotation, drop the rest
        if self.single_angle and self.rotation is not None:
            assert self.masks.shape[0] > 0, "Error! Must have masks to determine correct orientation of all letters."
            y = get_orientations(self.masks)
            inds = (y == self.rotation)
            self.data = self.data[inds]
            self.masks = self.masks[inds]
            self.size = len(self.data)

        # pad batch with zeros if necessary
        if pad:
            csz = self.size
            sz = csz + (self.batch_size - csz % self.batch_size )
            psz = [sz] + list(self.masks.shape[1:])

            if self.masks.shape[0] > 0:
                zmasks = np.cast[self.masks.dtype](np.zeros(psz))
                zmasks[:csz] = self.masks
                self.masks = zmasks

            zdata = np.cast[self.data.dtype](np.zeros(psz))
            zdata[:csz] = self.data
            self.data = zdata

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
            if self.masks.shape[0] > 0:
                self.masks = self.masks[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        if self.masks.shape[0] > 0:
            m = self.masks[self.p : self.p + n]
        else:
            m = self.masks
        self.p += self.batch_size
        
        return x.copy(),m.copy()
    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)