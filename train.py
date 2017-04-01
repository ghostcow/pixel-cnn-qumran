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
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet|letters')
parser.add_argument('-t', '--gen_interval', type=int, default=20, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
parser.add_argument('-f', '--rotation', type=int, default=None, help='Force uniform rotation of angle n*90 degrees counter-clockwise.')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=12, help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
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

# initialize data loaders for train/test splits
if args.data_set == 'imagenet' and args.class_conditional:
    raise("We currently don't have labels for the small imagenet data set")
DataLoader = {'cifar':cifar10_data.DataLoader, 'imagenet':imagenet_data.DataLoader, 'letters':letters_data.DataLoader}[args.data_set]
train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional, rotation=args.rotation)
test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional, rotation=args.rotation if args.rotation else 0)
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
#    _,y_sample,_ = test_data.next()
#    test_data.reset()
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
loss_gen_test = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # train
        gen_par = model(xs[i], hs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
        loss_gen.append(nn.discretized_mix_logistic_loss(xs[i], gen_par))
        # gradients
        grads.append(tf.gradients(loss_gen[i], all_params))
        # test
        gen_par = model(xs[i], hs[i], ema=ema, dropout_p=0., **model_opt)
        loss_gen_test.append(nn.discretized_mix_logistic_loss(xs[i], gen_par))

# add losses and gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1,args.nr_gpu):
        loss_gen[0] += loss_gen[i]
        loss_gen_test[0] += loss_gen_test[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    # training op
    optimizer = tf.group(nn.adam_updates(all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)

# convert loss to bits/dim
bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
bits_per_dim_test = loss_gen_test[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)

# sample from the model
new_x_gen = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        gen_par = model(xs[i], h_sample[i], ema=ema, dropout_p=0, **model_opt)
        new_x_gen.append(nn.sample_from_discretized_mix_logistic(gen_par, args.nr_logistic_mix))
def sample_from_model(sess, data, s=0):
    x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32) for i in range(args.nr_gpu)]
#    x_gen, labels, masks, k = data
#    x_gen = np.cast[np.float32]((x_gen - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
#    x_sample = np.rot90(x_gen[s], k=-k, axes=(0,1)).copy()
#    m_sample = np.rot90(masks[s], k=-k, axes=(0,1)).copy()
#    
#    label = labels[s]


#    x_gen[:,...] = x_sample
#    masks[:,...] = m_sample
#    for i,k in enumerate(np.concatenate(y_sample, axis=0)):
#        if i==0:
#            continue
#        x_gen[i] = np.rot90(x_gen[i], k=k, axes=(0,1))
#        masks[i] = np.rot90(masks[i], k=k, axes=(0,1))
#    x_gen = np.split(x_gen, args.nr_gpu)
#    masks = np.split(masks, args.nr_gpu)
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            new_x_gen_np = sess.run(new_x_gen, {xs[i]: x_gen[i] for i in range(args.nr_gpu)})
            for i in range(args.nr_gpu):
                x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
#                x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]*(1-masks[i][:,yi,xi,:])+x_gen[i][:,yi,xi,:]*masks[i][:,yi,xi,:]
    return np.concatenate(x_gen, axis=0)
#    x_gen = np.concatenate(x_gen, axis=0)
#    masks = np.concatenate(masks, axis=0)
    # color black fill-in as red, white fill-in as green
#    black_inds = ( (x_gen)*(1-masks) == -1 )
#    white_inds = np.cast[np.bool]( (~black_inds)*(1-masks) )
#    x_gen[black_inds[...,0],0] = 1
#    x_gen[white_inds[...,0],0] = -1
#    x_gen[white_inds[...,2],2] = -1
    # first sample is the real sample
#    x_gen[0] = x_sample
    # rotate back
#    x_gen = np.concatenate(x_gen, axis=0)
#    x_gen_unrot = x_gen.copy()
#    for i,k in enumerate(np.concatenate(y_sample, axis=0)):
#        if i==0:
#            continue
#        x_gen_unrot[i] = np.rot90(x_gen_unrot[i], k=-k, axes=(0,1))
#    return x_gen, x_gen_unrot, label


# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

# turn numpy inputs into feed_dict for use with tensorflow
def make_feed_dict(data, init=False):
    if type(data) is tuple and len(data)==2:
        x,m = data
        y = None
    elif type(data) is tuple and len(data)==3:
        x,y,m = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    
    # TODO: FIX THIS!!!! (randomly) rotate batch 
    if args.rotation is None:
        k = np.random.randint(4)
        y.fill(k)
    else:
        k = args.rotation
    x = np.rot90(x, k=k, axes=(1,2))
    m = np.rot90(m, k=k, axes=(1,2))
    
    if init:
        feed_dict = {x_init: x}
        if y is not None and args.class_conditional:
            feed_dict.update({y_init: y})
    else:
        x = np.split(x, args.nr_gpu)
        feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
        if y is not None and args.class_conditional:
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
test_bpd = []
min_test_loss = np.inf
test_loss_gen = np.inf
lr = args.learning_rate
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

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
            print('initializing the model...')
            if not args.just_gen:
                feed_dict = make_feed_dict(train_data.next(args.init_batch_size), init=True) # manually retrieve exactly init_batch_size examples
                train_data.reset()  # rewind the iterator back to 0 to do one full epoch
                sess.run(initializer, feed_dict)
            elif args.load_params:
                ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
                print('restoring parameters from', ckpt_file)
                saver.restore(sess, ckpt_file)
            else:
                print("error, can't generate samples without loading params.")
                sys.exit(-1)
        
        if not args.just_gen:
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
    
            # compute likelihood over test data
            test_losses = []
            for d in test_data:
                feed_dict = make_feed_dict(d)
                l = sess.run(bits_per_dim_test, feed_dict)
                test_losses.append(l)
            test_loss_gen = np.mean(test_losses)
            test_bpd.append(test_loss_gen)
    
            # log progress to console
            print("Iteration %d, time = %ds, train bits_per_dim = %.4f, test bits_per_dim = %.4f" % (epoch, time.time()-begin, train_loss_gen, test_loss_gen))
            sys.stdout.flush()
                   
        if epoch % args.gen_interval == 0 and epoch > 0:
            # generate samples from the model
            print('generating samples from model...')
            data = test_data.next()
            test_data.reset()

            def print_samples(sample_x, suffix=''):
                img_tile = plotting.img_tile(sample_x[:int(np.floor(np.sqrt(args.batch_size*args.nr_gpu))**2)], aspect_ratio=1.0, border_color=1.0, stretch=True).squeeze()
                img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
                plotting.plt.savefig(os.path.join(args.save_dir,'%s_sample%d%s.png' % (args.data_set, epoch, suffix)))
                plotting.plt.close('all')

            sample_x = sample_from_model(sess, data)
            print_samples(sample_x)
            print('done.')
            
        # save params via early stopping
        current = test_loss_gen

        if np.less(current - min_delta, best):
            best = current
            wait = 0
            print('Saving model params...', end='')
            saver.save(sess, args.save_dir + '/params_' + args.data_set + '.ckpt')
            np.savez(args.save_dir + '/test_bpd_' + args.data_set + '.npz', test_bpd=np.array(test_bpd))
            print('done.')
        else:
            if wait >= patience:
                stopped_epoch = epoch
                print('Epoch %05d: early stopping' % (stopped_epoch))
                stop_training = True
            wait += 1
            
        if stop_training:
            break