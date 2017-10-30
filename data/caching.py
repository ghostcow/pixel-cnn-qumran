import scipy.misc as sm
import numpy as np
from glob import glob
import pickle as pkl
import os

def parse_resize(imgs, s=32):
    img_rsz = np.zeros((len(imgs),s,s,3), dtype=np.uint8)
    c=s//2
    for j,im in enumerate(imgs):
        h,w = im.shape[0],im.shape[1]
        if h>s:
            ar = s/h
            im = sm.imresize(im, ar, interp='bicubic')
            h, w = im.shape[0], im.shape[1]
        if w>s:
            ar = s/w
            im = sm.imresize(im,ar,interp='bicubic')
            h, w = im.shape[0], im.shape[1]
        img_rsz[j,(c-h//2):(c+h//2+h%2),(c-w//2):(c+w//2+w%2),:]=im
    return img_rsz

def get_image_tensor(file_list):
    imgs = [sm.imread(file, mode='RGB') for file in file_list]
    return parse_resize(imgs)

def build_cache(path):
    return sorted(glob(os.path.join(path, 'images', '*.jpg')))

def load_cache(path):
    cache_file = os.path.join(path, 'cache.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pkl.load(f)
    else:
        return None

def save_cache(path, file_list, data):
    cache_file = os.path.join(path, 'cache.pkl')
    with open(cache_file, 'wb') as f:
        pkl.dump({
            'file_list': file_list,
            'images': data,
            'masks': None
        }, f)

def load_data(path):
    file_list = build_cache(path)
    cache = load_cache(path)
    if cache is None or set(cache['file_list']) != set(file_list):
        images = get_image_tensor(file_list)
        save_cache(path, file_list, images)
        return data
    else:
        return cache['images']