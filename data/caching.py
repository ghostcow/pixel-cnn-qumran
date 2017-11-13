import scipy.misc as sm
import numpy as np
from glob import glob
import pickle as pkl
import os

def parse_resize(imgs, s=32):
    img_rsz = np.zeros((len(imgs),s,s,1), dtype=np.uint8)
    c=s//2
    for j,im in enumerate(imgs):
        h,w = im.shape[0],im.shape[1]
        if h>s:
            ar = s/h
            im = sm.imresize(im, ar, interp='bicubic')
            h, w = im.shape[0], im.shape[1]
        if w>s:
            ar = s/w
            im = sm.imresize(im, ar, interp='bicubic')
            h, w = im.shape[0], im.shape[1]
        im = im[...,0,np.newaxis] # convert to grayscale
        img_rsz[j,(c-h//2):(c+h//2+h%2),(c-w//2):(c+w//2+w%2),:]= im
    return img_rsz

def get_image_list(path):
    return sorted(glob(os.path.join(path, '*.jpg')) +
                  glob(os.path.join(path, '*.png')))

def get_image_tensor(image_list):
    # assuming masks and letters are stored as RGB images
    imgs = [sm.imread(file, mode='RGB') for file in image_list]
    # we convert to Grayscale upon resizing
    return parse_resize(imgs)

def load_cache(path):
    cache_file = os.path.join(path, 'cache.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pkl.load(f)
    else:
        return None

def save_cache(path, letter_list, letter_tensor, mask_list, mask_tensor):
    cache_file = os.path.join(path, 'cache.pkl')
    with open(cache_file, 'wb') as f:
        pkl.dump({
            'letter_list': letter_list,
            'letter_tensor': letter_tensor,
            'mask_list': mask_list,
            'mask_tensor': mask_tensor
        }, f)

'''
The data is structured as follows:
data/letters - place the output of the letter spotting algorithm here
data/masks - (optional) place the masks you created for the data here

If there are no masks specified, randomly generate half-planes with a quarter missing as masks
'''
def load_data(path):
    letter_list = get_image_list(os.path.join(path, 'letters'))
    mask_list = get_image_list(os.path.join(path, 'masks'))
    cache = load_cache(path)
    if cache is None \
            or set(cache['letter_list']) != set(letter_list) \
            or set(cache['mask_list']) != set(mask_list):

        letter_tensor = get_image_tensor(letter_list)
        mask_tensor = get_image_tensor(mask_list) // 255 # normalize to {0,1} values
        save_cache(path, letter_list, letter_tensor, mask_list, mask_tensor)
    else:
        letter_tensor = cache['letter_tensor']
        mask_tensor = cache['mask_tensor']

    return letter_tensor, mask_tensor