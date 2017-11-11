"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr-gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

import os
import sys
import time
import json
import argparse
from tqdm import tqdm

import numpy as np
import tensorflow as tf

import pixel_cnn_pp.nn as nn
import pixel_cnn_pp.plotting as plotting
from pixel_cnn_pp.model import model_spec
import data.cifar10_data as cifar10_data
import data.imagenet_data as imagenet_data
import data.letters_data as letters_data

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='/tmp/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/tmp/pxpp/save', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='letters', help='Currently supports only letters')
parser.add_argument('-t', '--gen_interval', type=int, default=20, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
parser.add_argument('-f', '--rotation', type=int, default=None, help='Force uniform rotation of angle n*90 degrees counter-clockwise.')
parser.add_argument('-u', '--randomize_labels', dest='randomize_labels', action='store_true', help='Randomize labels')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=12, help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
parser.add_argument('--gpu_mem_frac', type=float, default=1.0, help='Limit GPU memory to this fraction of itself during session')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('-j', '--just_gen', dest='just_gen', action='store_true', help='Just generate samples without training.')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# initialize data loaders for train/val splits
DataLoader = {'letters':letters_data.DataLoader}[args.data_set]
train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional, rotation=args.rotation)
val_data = DataLoader(args.data_dir, 'val', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional, rotation=args.rotation if args.rotation else 0)
obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
assert len(obs_shape) == 3, 'assumed right now'

# data place holders
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape) for i in range(args.nr_gpu)]

# if the model is class-conditional we'll set up label placeholders + one-hot encodings 'h' to condition on
if args.class_conditional:
    num_labels = train_data.get_num_labels()
    y_init = tf.placeholder(tf.int32, shape=(args.init_batch_size,))
    h_init = tf.one_hot(y_init, num_labels)
    y_sample = np.arange(args.batch_size * args.nr_gpu) % num_labels
    y_sample = np.split(y_sample, args.nr_gpu)
    h_sample = [tf.one_hot(tf.Variable(y_sample[i], trainable=False), num_labels) for i in range(args.nr_gpu)]
    ys = [tf.placeholder(tf.int32, shape=(args.batch_size,)) for i in range(args.nr_gpu)]
    hs = [tf.one_hot(ys[i], num_labels) for i in range(args.nr_gpu)]
else:
    h_init = None
    h_sample = [None] * args.nr_gpu
    hs = h_sample

# create the model
model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity }
model = tf.make_template('model', model_spec)

# run once for data dependent initialization of parameters
gen_par = model(x_init, h_init, init=True, dropout_p=args.dropout_p, **model_opt)

# keep track of moving average
all_params = tf.trainable_variables()
ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
maintain_averages_op = tf.group(ema.apply(all_params))

# get loss gradients over multiple GPUs
grads = []
loss_gen = []
loss_gen_val = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # train
        gen_par = model(xs[i], hs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
        loss_gen.append(nn.discretized_mix_logistic_loss(xs[i], gen_par))
        # gradients
        grads.append(tf.gradients(loss_gen[i], all_params))
        # val
        gen_par = model(xs[i], hs[i], ema=ema, dropout_p=0., **model_opt)
        loss_gen_val.append(nn.discretized_mix_logistic_loss(xs[i], gen_par))

# add losses and gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1,args.nr_gpu):
        loss_gen[0] += loss_gen[i]
        loss_gen_val[0] += loss_gen_val[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    # training op
    optimizer = tf.group(nn.adam_updates(all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)

# convert loss to bits/dim
bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
bits_per_dim_val = loss_gen_val[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)

# sample from the model
new_x_gen = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        gen_par = model(xs[i], h_sample[i], ema=ema, dropout_p=0, **model_opt)
        new_x_gen.append(nn.sample_from_discretized_mix_logistic(gen_par, args.nr_logistic_mix))
def sample_from_model(sess):
    x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32) for i in range(args.nr_gpu)]
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            new_x_gen_np = sess.run(new_x_gen, {xs[i]: x_gen[i] for i in range(args.nr_gpu)})
            for i in range(args.nr_gpu):
                x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
    return np.concatenate(x_gen, axis=0)


# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

def flip_rotate(x, y):
    '''
    flips and/or rotates a single image according to label y
    y is encoded to represent flips and rotations
    y \in {0..7} or y \in {-7..-1} for reversing rotations\flips
    flip indicator = y // 4 ( or y <= -4 in the negative case )
    rotation angle = (y % 4) * 90 degrees
    because of dihedral group D4 structure, in some cases the order of flip/rotation
    matters, and is dealt with accordingly.
    '''
    if y // 4 == 1 or y == -4 or y == -6:
        x = np.flip(x, len(x.shape)-2)
    if len(x.shape) == 4:
        x = np.rot90(x, k= y % 4, axes=(1,2))
    else:
        x = np.rot90(x, k= y % 4)
    if y == -5 or y == -7:
        x = np.flip(x, len(x.shape)-2)
    return x

def adaptive_rotation(x, y):
    if y is None:
        pass
    elif type(y)==int:
        x = flip_rotate(x, y)
    else:
        for j in range(len(y)):
            x[j] = flip_rotate(x[j], y[j])
    return x

# turn numpy inputs into feed_dict for use with tensorflow
def make_feed_dict(data, init=False, val=False):
    '''
    contract: class_conditional => randomize_labels must be at the same time
    '''
    if type(data) is tuple and len(data)==2:
        x,m = data
        y = None
        x = adaptive_rotation(x, args.rotation)

    # randomize labels by selecting one random label per batch, unrotate and 
    # unflip if necessary, turn off on validation
    if args.randomize_labels and val is not True:
        x = adaptive_rotation(x, -args.rotation)
        y = np.zeros(x.shape[0], dtype=np.int32)
        y.fill(np.random.randint(8))
        x = adaptive_rotation(x, y)

    x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    
    if init: # we don't call init=True with val=True...
        if args.randomize_labels: # in case y is some const
            x = adaptive_rotation(x, -y)
            y = np.arange(x.shape[0]) % 8
            x = adaptive_rotation(x, y)
        feed_dict = {x_init: x}
        if args.class_conditional:
            feed_dict.update({y_init: y})
    else:
        x = np.split(x, args.nr_gpu)
        feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
        if args.class_conditional:
            y = np.zeros(args.batch_size * args.nr_gpu, dtype=np.int32)
            y.fill(args.rotation)
            y = np.split(y, args.nr_gpu)
            feed_dict.update({ys[i]: y[i] for i in range(args.nr_gpu)})
    return feed_dict

# //////////// perform training //////////////
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    
# save hyperparms to file
with open(os.path.join(args.save_dir,'hyperparams.txt'),'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',',':')))
print('starting training')
val_bpd = []
min_val_loss = np.inf
val_loss_gen = np.inf
lr = args.learning_rate

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = max(min(args.gpu_mem_frac, 1.0), 0.0)

# early stopping params
patience = 150
min_delta = 0
min_delta *= -1
wait = 0
stopped_epoch = 0
best = np.Inf
stop_training = False

with tf.Session(config=config) as sess:
    for epoch in range(args.max_epochs):
        begin = time.time()
        
        # init
        if epoch == 0:
            # manually retrieve exactly init_batch_size examples
            feed_dict = make_feed_dict(
                train_data.next(args.init_batch_size), init=True)
            train_data.reset()  # rewind the iterator back to 0 to do one full epoch
            sess.run(initializer, feed_dict)
            print('initializing the model...')
            if args.load_params:
                ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
                print('restoring parameters from', ckpt_file)
                saver.restore(sess, ckpt_file)

        # train for one epoch
        train_losses = []
        for d in tqdm(train_data):
            feed_dict = make_feed_dict(d)
            # forward/backward/update model on each gpu
            lr *= args.lr_decay
            feed_dict.update({ tf_lr: lr })
            l,_ = sess.run([bits_per_dim, optimizer], feed_dict)
            train_losses.append(l)
        train_loss_gen = np.mean(train_losses)

        # compute likelihood over val data
        val_losses = []
        for d in val_data:
            feed_dict = make_feed_dict(d, val=True)
            l = sess.run(bits_per_dim_val, feed_dict)
            val_losses.append(l)
        val_loss_gen = np.mean(val_losses)
        val_bpd.append(val_loss_gen)

        # log progress to console
        print("Iteration %d, time = %ds, train bits_per_dim = %.4f, val bits_per_dim = %.4f" % (epoch, time.time()-begin, train_loss_gen, val_loss_gen))
        sys.stdout.flush()
                   
        if epoch % args.gen_interval == 0 and epoch > 0:
            # generate samples from the model
            print('generating samples from model...')
            def print_samples(sample_x, suffix=''):
                img_tile = plotting.img_tile(sample_x[:int(np.floor(np.sqrt(args.batch_size*args.nr_gpu))**2)], aspect_ratio=1.0, border_color=1.0, stretch=True).squeeze()
                plotting.plot_img(img_tile, title=args.data_set + ' samples')
                plotting.plt.savefig(os.path.join(args.save_dir,'%s_sample%d%s.png' % (args.data_set, epoch, suffix)))
                plotting.plt.close('all')
            sample_x = sample_from_model(sess)
            print_samples(sample_x)
            print('done.')
            
        # save params via early stopping
        current = val_loss_gen
        current_train = train_loss_gen

        if np.less(current - min_delta, best):
            best = current
            best_train = current_train
            wait = 0
            print('Saving model params...', end='')
            saver.save(sess, args.save_dir + '/params_' + args.data_set + '.ckpt')
            np.savez(args.save_dir + '/val_bpd_' + args.data_set + '.npz', val_bpd=np.array(val_bpd))
            print('done.')
        else:
            if wait >= patience:
                stopped_epoch = epoch
                print('Epoch %05d: early stopping' % (stopped_epoch))
                print("Best iteration: %d, train bits_per_dim = %.4f, val bits_per_dim = %.4f" % (stopped_epoch-patience, best_train, best))
                stop_training = True
            wait += 1
            
        if stop_training:
            break